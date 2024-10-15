import jittor as jt
import jittor.nn as nn
import numpy as np

from jnerf.models.samplers.neuralto_render import MipSSS

def smithG1(cosTheta, alpha):
    sinTheta = jt.sqrt(1.0 - cosTheta * cosTheta)
    tanTheta = sinTheta / (cosTheta + 1e-10)
    root = alpha * tanTheta
    return 2.0 / (1.0 + jt.hypot(root, jt.ones_like(root)))

class MaterialPredictor(nn.Module):
    def __init__(self, sdf_network, material_network):
        super().__init__()
        self.sdf_network = sdf_network
        self.material_network = material_network

    def execute(self, points):
        _, features, normals = self.sdf_network.get_all(points, is_training=False)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-10)
        result = self.material_network(
            points, normals, features
        )
        diffuse_albedo = result["diffuse_albedo"]
        trans_albedo = result["trans_albedo"]
        specular_roughness = result["specular_roughness"]
        return diffuse_albedo, trans_albedo, specular_roughness


class NeuralTOScatterRenderer(nn.Module):

    def __init__(self, encoder_cfg, add_ss=False,
                 nerf_num_samples=32,
                 mipmap_scale=0.8,
                 mipmap_level=4,
                 sphere_num_samples=64,
                 predict_sigma=False,
                 predict_color=True):
        # light model and render model
        super().__init__()
        self.render_model = None
        # This specified param is For Testing Only!
        self.laplace_beta = 8.067957423918415e-06
        self.laplace_factor = 5.0610198974609375
        # self.laplace_beta = 0.0008
        # self.laplace_factor = 5.0
        # scale factor for unit sphere
        self.dist = 1.0
        self.r = 1.0
        self.ior = 1.25
        self.add_ss = add_ss
        self.tone_mapping = None
        self.render_model = MipSSS(encoder_cfg,
                                   nerf_num_samples,
                                   mipmap_scale,
                                   mipmap_level,
                                   sphere_num_samples,
                                   predict_sigma,
                                   predict_color)
    
    def network_setup(self, scatter_model):
        self.render_model = scatter_model
   
    def execute(self, light_model, sdf_network,
                diffuse_albedo, trans_albedo, specular_roughness,
                surf_points, surf_normal, surf_distance, viewdir,
                thickness, light_o, conv_mask, rays_t):
        # compute light
        # ==============
        light = light_model() * self.dist * self.dist
        # render specular
        # =================
        light_intensity = light / (surf_distance * surf_distance + 1e-10)
        dot = jt.sum(viewdir * surf_normal, dim=-1, keepdims=True)
        dot = jt.clamp(dot, min_v=0.00001, max_v=0.99999)  # must be very precise; cannot be 0.999
        alpha = specular_roughness
        cosTheta2 = dot * dot
        root = cosTheta2 + (1.0 - cosTheta2) / (alpha * alpha + 1e-10)
        D = 1.0 / (np.pi * alpha * alpha * root * root + 1e-10)
        # F = fr_schlick(self.ior, dot)
        F = 0.03867
        G = smithG1(dot, alpha) ** 2  # [..., 1]

        # pre-defined specular_albedo=0.5
        specular_color = 0.5 * light_intensity * F * D * G / (4.0 * dot + 1e-10)
        # render diffuse
        # ================
        # light_o = surf_points + surf_distance * (-rays_t)
        light_I = trans_albedo * light_intensity * (1 - F)
        # light_I_ori = light * jt.ones_like(light_intensity)
        single_s, multi_s = self.render_model(base_r=self.r,
                                              surface_distance=surf_distance,
                                              rays_o=surf_points + thickness / 64,
                                              rays_d=rays_t,
                                              light_I=light_I,
                                              # light_I_ori=light_I_ori,
                                              light_o=light_o,
                                              distance=thickness,
                                              sdf_network=sdf_network,
                                              beta=self.laplace_beta,
                                              factor=self.laplace_factor,  # [p,1]
                                              diffuse_albedo=diffuse_albedo)

        # Fake ss
        Fss90 = 1 * 1 * alpha  # cos(half-vector, wi)*cos(half-vector, wo)*alpha
        Fss = (1 + (Fss90 - 1) * F) * (1 + (Fss90 - 1) * F)
        dot_abs = jt.abs(jt.sum(viewdir * surf_normal, dim=-1, keepdims=True))
        S = 1.25 * (Fss * (1 / (dot_abs + dot_abs) - 0.5) + 0.5)
        fr = S / np.pi
        surf_color = diffuse_albedo * fr * (light_intensity * dot)

        surf_color_trans = surf_color * (1 - trans_albedo)
        # single_color = single_s * diffuse_albedo
        # trans_color = multi_s * diffuse_albedo

        diffuse_color = surf_color_trans + multi_s + (single_s * dot)
        
        results = {
            "diffuse_rgb": diffuse_color,
            "specular_rgb": specular_color,
            "rgb": diffuse_color + specular_color,
            "multi_ss": multi_s,
            "single_ss": single_s,
            "single_color": (single_s * dot),
        }
        return results
