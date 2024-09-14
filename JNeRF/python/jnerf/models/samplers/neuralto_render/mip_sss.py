import jittor as jt
from jittor import nn
import numpy as np

from jnerf.models.samplers.neuralto_render.recon_renderer import laplace
from jnerf.utils.miputils import conical_frustum_to_gaussian, integrated_pos_enc
from jnerf.utils.sh import HardCodedSH
from jnerf.models.position_encoders.freq_encoder import FrequencyEncoder

def standard_fibonacci_sampling_np(num_samples):
    """
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(num_samples):
        y = 1 - (i / float(num_samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points, dtype=np.float32)

    return points

def dot_product(a, b):
    return jt.sum(jt.mul(a, b), dim=-1, keepdim=True)


def normalize(a):
    return jt.div(a, a.norm(dim=-1, keepdim=True) + 1e-10)

class MipSSS(nn.Module):

    def __init__(self,
        nerf_num_samples=32,
        mipmap_scale=0.8,
        mipmap_level=4,
        sphere_num_samples=64,
        predict_sigma=False,
        predict_color=True,
        dirs_multires=2,
        points_multires=10
    ):
        super().__init__()
        self.nerf_num_samples = nerf_num_samples
        self.mipmap_scale = 0.8  # 2 / sqrt(12)
        self.mipmap_level = mipmap_level
        self.sh_band = 3
        self.sphere_num_samples = sphere_num_samples
        self.predict_sigma = predict_sigma
        self.predict_color = predict_color

        self.feature_dim = 64
        self.sca_feature_dim = 64

        self.dirs_input_dim = 3
        self.dirs_multires = dirs_multires

        self.points_input_dim = 3
        self.points_multires = points_multires

        # ========= original nerf
        self.density_activation = nn.Sequential(*[
            nn.Linear(self.feature_dim+1, 1),
            jt.nn.Softplus(),
        ])
        self.relu = nn.ReLU()

        self.l_embedder = FrequencyEncoder(
            multires=self.dirs_multires,
            input_dims=3,
            include_input=False)
        self.l_input_dim = self.l_embedder.out_dim
        self.p_embedder = FrequencyEncoder(
            multires=self.points_multires,
            input_dims=3,
            include_input=False)
        self.points_input_dim = self.p_embedder.out_dim

        self.F_dim = self.points_input_dim
        feature_mlp_layers = [nn.Linear(self.F_dim, self.feature_dim)]
        for i in range(3):
            feature_mlp_layers.append(nn.Linear(self.feature_dim, self.feature_dim))
            feature_mlp_layers.append(self.relu)
        self.feature_mlp = nn.Sequential(*feature_mlp_layers)
        # ==================================

        # Light network
        self.color_input_dim = self.points_input_dim + self.l_input_dim + 3  # L(L0, x, n)
        color_mlp_layers = [nn.Linear(self.color_input_dim, self.feature_dim)]
        for i in range(2):
            color_mlp_layers.append(nn.Linear(self.feature_dim, self.feature_dim))
            feature_mlp_layers.append(self.relu)
        color_mlp_layers.extend([
            nn.Linear(self.feature_dim, 3),
            nn.Sigmoid()
        ])
        self.color_mlp = nn.Sequential(*color_mlp_layers)

        # scattering residual network
        self.bridge_block = nn.Sequential(*[
            nn.Linear(self.points_input_dim + self.sca_feature_dim, self.sca_feature_dim, bias=False)
        ])
        self.bridge_block_0 = nn.Linear(self.points_input_dim+3+self.l_input_dim, self.sca_feature_dim)
        residual_block = []
        for i in range(5):
            residual_block.append(nn.Linear(self.sca_feature_dim, self.sca_feature_dim))
            residual_block.append(self.relu)
        self.residual_block_mlp = nn.Sequential(*residual_block)

        # sh network
        self.num_coeff = (self.sh_band + 1) ** 2
        self.num_coeff *= 3
        self.sh = HardCodedSH(self.sh_band)

        self.weight_predict_block = nn.Sequential(*[
            nn.Linear(self.sca_feature_dim, self.sca_feature_dim),
            self.relu,
            nn.Linear(self.sca_feature_dim, self.num_coeff),
        ])

        self.raw2alpha = lambda raw, dists: 1. - jt.exp(-raw * dists)
        # self.laplace = lambda sdf, beta: 1. * (0.5 + 0.5 * sdf.sign() * jt.expm1(-sdf.abs() / beta))

        self.sh_sampled_dirs = jt.array(
            standard_fibonacci_sampling_np(self.sphere_num_samples))  # [sphere_num_samples, 3]
        self.sh_sampled_dirs = normalize(self.sh_sampled_dirs)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def execute(self, base_r, distance, rays_o, rays_d, light_I, light_o, surface_distance, sdf_network,
                beta, factor, diffuse_albedo):

        num_rays = rays_o.shape[0]

        # sample mid points
        t_samples = jt.linspace(0., 1., self.nerf_num_samples)
        t_samples = jt.broadcast(t_samples, [num_rays, self.nerf_num_samples])
        t_samples = distance * t_samples  # use real values of t instead of proportions of far
        pts = rays_o[..., None, :] + rays_d[..., None, :] * t_samples[..., :, None]  # [num_rays, num_samples, 3]

        # geo information: SDF normal
        pts_flatten = pts.view(-1, 3)
        _y, _feature, gradients = sdf_network.get_all(pts_flatten, is_training=False)

        # sdf_normals = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-7)
        # sdf_normals = sdf_normals.reshape(num_rays, self.nerf_num_samples, 3)
        grad = gradients.reshape(num_rays, self.nerf_num_samples, 3)
        sigma = factor * laplace(_y, beta+0.0001)
        sigma = sigma.reshape(num_rays, self.nerf_num_samples, 1)

        # predict density
        level0_enc = self.p_embedder(pts)
        if self.predict_sigma:
            feat = self.feature_mlp(level0_enc)
            density = self.density_activation(jt.concat([feat, sigma], dim=-1)).squeeze(-1)
        else:
            density = sigma.squeeze(-1)

        input_light = light_I[:, None, :].repeat(1, self.nerf_num_samples, 1)

        dists = t_samples[..., 1:] - t_samples[..., :-1]
        dists = jt.concat([dists, jt.array([1e10]).expand(dists[..., :1].shape)], -1)
        alpha = self.raw2alpha(density, dists)
        transmittance = jt.cumprod(
            jt.concat([jt.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

        weights = alpha * transmittance
        # Ls = phase_HG(g, l_dot_o) * input_light * transmittance[..., None] * albedo

        L = input_light * transmittance[..., None]
        L = L.repeat(1, 1, 3)
        Ls = L / (4 * np.pi)

        duplicate_rays_d = rays_d[:, None, :].repeat(1, self.nerf_num_samples, 1)
        rays_d_feat = self.l_embedder(duplicate_rays_d)

        if self.predict_color:
            albedo = self.color_mlp(jt.concat([level0_enc, rays_d_feat, grad], dim=-1))
        else:
            albedo = diffuse_albedo[:, None, :].repeat(1, self.nerf_num_samples, 1)

        # multi scattering
        t_samples_mips = jt.linspace(0., 1., self.nerf_num_samples)
        t_samples_mips = surface_distance + distance * t_samples_mips
        t_samples_mips = jt.broadcast(t_samples_mips, [num_rays, self.nerf_num_samples])
        Lmf = None

        for i in range(self.mipmap_level):
            if i == 0:
                Lmf = self.bridge_block_0(jt.concat([level0_enc, L, rays_d_feat], dim=-1))
            else:
                radii = base_r / pow(self.mipmap_scale, i)
                h = base_r / pow(self.mipmap_scale, i)

                t0 = t_samples_mips - h/2  # [B, n_samples]
                t1 = t_samples_mips + h/2
                means, covs = conical_frustum_to_gaussian(rays_d, t0, t1, radii, diag=True)
                means = means + jt.unsqueeze(light_o, dim=-2)
                leveli_enc = integrated_pos_enc(
                    (means, covs),
                    min_deg=0,
                    max_deg=self.points_multires,
                )  # samples_enc: [B, N, 2*3*L]  L:(max_deg_point - min_deg_point)
                Lmf = self.bridge_block(jt.concat([leveli_enc, Lmf], dim=-1))
            Lmf = self.residual_block_mlp(Lmf) + Lmf

        sh_weights = self.weight_predict_block(Lmf)  # [num_rays, num_nerf_samples, ch]
        sh_sampled_d = self.sh_sampled_dirs.unsqueeze(0).repeat(num_rays, 1, 1)
        sh_radiance = self.sh.eval_sh_bases(sh_sampled_d)  # [num_rays, num_sh_samples, ch]
        sh_radiance = sh_radiance.permute(0, 2, 1)  # [num_rays, ch, num_sh_samples]

        Lm = jt.zeros_like(L)
        for i, sh_w in enumerate(jt.split(sh_weights, 16, dim=-1)):
            S = jt.clamp(nn.bmm(sh_w, sh_radiance))
            Lm[:, :, i] = jt.mean(S, dim=-1)

        Lm = Lm * albedo
        Ls = Ls * albedo
        single_ss = jt.sum(weights[..., None] * Ls, -2)  # [N_rays, 3]
        multi_ss = jt.sum(weights[..., None] * Lm, -2)

        return single_ss, multi_ss