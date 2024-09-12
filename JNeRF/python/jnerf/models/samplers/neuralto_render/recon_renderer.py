import jittor as jt
import jittor.nn as nn

import numpy as np
import logging
import mcubes
from jnerf.models.samplers.neus_render.renderer import extract_fields, extract_geometry, sample_pdf
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import SAMPLERS


def expm1(value):
    return jt.exp(value) - 1


def laplace(sdf, beta):
    # Seems that jittor does not has the function expm1, which provides greater precision than exp(x) - 1 for small values of x
    # res = torch.where(x >= 0, 0.5 * torch.exp(-x / beta), 1 - 0.5 * torch.exp(x / beta))
    beta_ = beta + 0.00001  # min of beta
    # alpha = 1 / beta_
    return 1 * (0.5 + 0.5 * nn.sign(sdf) * expm1(-sdf.abs() / beta_))


# Finding nearest and farest intersection points
# ===============================================
def find_near_intersection(sdf_val, z_val, near):
    num_rays, num_samples = sdf_val.shape
    tmp = nn.sign(sdf_val) * jt.arange(num_samples, 0, -1).reshape([1, num_samples])
    # return first negative sdf point if exists
    min_val, min_idx = jt.arg_reduce(tmp, 'min', dim=-1, keepdims=False)
    near_low = near.clone()
    intersection_work_mask = (min_val < 0.0) & (min_idx >= 1)
    n_intersection = intersection_work_mask.sum()
    if n_intersection > 0:
        min_idx = jt.unsqueeze(min_idx[intersection_work_mask], -1)
        z_low = jt.gather(z_val[intersection_work_mask], dim=-1, index=min_idx - 1)
        sdf_low = jt.gather(sdf_val[intersection_work_mask], dim=-1, index=min_idx - 1)
        z_high = jt.gather(z_val[intersection_work_mask], dim=-1, index=min_idx)  # [n_rootfind, ]
        # [n_rootfind, 1] < 0
        sdf_high = jt.gather(sdf_val[intersection_work_mask], dim=-1, index=min_idx)
        t = (sdf_low * z_high - sdf_high * z_low) / (sdf_low - sdf_high)
        near_low[intersection_work_mask] = t

    return near_low


def find_far_intersection(sdf_val, z_val, far):
    num_rays, num_samples = sdf_val.shape
    tmp = nn.sign(sdf_val) * jt.arange(0, num_samples, 1).reshape([1, num_samples])
    min_val, min_idx = jt.arg_reduce(tmp, 'min', dim=-1, keepdims=False)
    far_high = far.clone()
    intersection_work_mask = (min_val < 0.0) & (min_idx < num_samples - 1)
    n_intersection = intersection_work_mask.sum()
    if n_intersection > 0:
        min_idx = jt.unsqueeze(min_idx[intersection_work_mask], -1)
        z_low = jt.gather(z_val[intersection_work_mask], dim=-1, index=min_idx + 1)
        sdf_low = jt.gather(sdf_val[intersection_work_mask], dim=-1, index=min_idx + 1)
        z_high = jt.gather(z_val[intersection_work_mask], dim=-1, index=min_idx)  # [n_rootfind, ]
        # [n_rootfind, 1] < 0
        sdf_high = jt.gather(sdf_val[intersection_work_mask], dim=-1, index=min_idx)
        t = (sdf_low * z_high - sdf_high * z_low) / (sdf_low - sdf_high)
        far_high[intersection_work_mask] = t
    return far_high


# =================================================

@SAMPLERS.register_module()
class NeuralTOReconRenderer:
    def __init__(self,
                 n_samples,
                 n_importance,
                 n_inner,
                 up_sample_steps,
                 perturb=-1,
                 weight_trans=0.5,
                 n_outside=0,
                 use_hf_weight=False,
                 inner_weight_norm=True):
        self.nerf = None
        self.sdf_network = None
        self.deviation_network = None
        self.color_network = None
        self.color_network_inner = None
        self.beta_network = None
        self.factor = None

        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.n_inner_samples = n_inner

        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.w_trans = weight_trans
        self.raw2alpha = lambda raw, dists: 1. - jt.exp(-raw * dists)

        self.use_hf_weight = use_hf_weight
        self.inner_weight_norm = inner_weight_norm

    def network_setup(self, network):
        self.nerf = network.nerf_outside
        self.sdf_network = network.sdf_network
        self.deviation_network = network.deviation_network
        self.color_network = network.color_network
        self.color_network_inner = network.color_network_inner
        self.factor = network.factor
        self.beta_network = network.beta_network

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.Var([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = jt.norm(pts, p=2, dim=-1, keepdim=True, eps=1e-6).safe_clip(1.0, 1e5)
        pts = jt.concat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        sampled_color = jt.sigmoid(sampled_color)
        alpha = 1.0 - jt.exp(-nn.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples).safe_clip(-1e6, 1e6)
        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-6], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
            'density': density.reshape(batch_size, n_samples),
            'dists': dists
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = jt.norm(pts, p=2, dim=-1, keepdim=False, eps=1e-6)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = jt.concat([jt.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = jt.stack([prev_cos_val, cos_val], dim=-1)
        cos_val = jt.min(cos_val, dim=-1, keepdims=False)
        cos_val = cos_val.safe_clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = jt.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = jt.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * jt.cumprod(
            jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-6], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = jt.concat([z_vals, new_z_vals], dim=-1)
        index, z_vals = jt.argsort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = jt.concat([sdf, new_sdf], dim=-1)
            xx = jt.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    z_vals_inner,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = jt.concat([dists, jt.Var([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        sdf_nn_output = self.sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = self.sdf_network.gradient(pts)
        # print(pts.shape)
        # print(gradients.shape)
        # print(dirs.shape)
        # print(feature_vector.shape)
        sampled_color = self.color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, -1)

        inv_s = self.deviation_network(jt.zeros([1, 3]))[:, :1].safe_clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        true_cos = (dirs * gradients).sum(-1, keepdims=True)
        true_cos = true_cos.safe_clip(-1, 1)
        iter_cos = -(nn.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     nn.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        if self.use_hf_weight:
            # another way for calculating NeuS's weight, we consider method in NeuS is a dispersed approaching
            # code in https://github.com/yiqun-wang/HFS/blob/main/models/renderer_high.py
            s = self.deviation_network(jt.zeros([1])).safe_clip(1e-6, 1e3)
            sigmoid_sdf = jt.sigmoid(-s * sdf)
            weight_sdf = s * sigmoid_sdf * (1 - sigmoid_sdf)
            weight_sdf = weight_sdf.reshape(batch_size, n_samples, 1) / (
                    weight_sdf.reshape(batch_size, n_samples, 1).sum(dim=1, keepdims=True) + 1e-6)

            # reassign weight value for SMALL one
            # 0.2 is a strange value
            weight_sdf[jt.squeeze(weight_sdf.sum(1)) < 0.2] = jt.unsqueeze(jt.ones([n_samples]), 1) / n_samples
            w_s = (gradients.norm(dim=1, keepdim=True).reshape(batch_size, n_samples, 1) * weight_sdf.detach()).sum(
                dim=1, keepdim=True)
            w_s = jt.exp(w_s - 1)
            inv_s = inv_s.expand(batch_size, n_samples, 1) * w_s
            inv_s = inv_s.reshape(batch_size * n_samples, 1)

            mid_dists = mid_z_vals[..., 1:] - mid_z_vals[..., :-1]
            mid_dists = jt.cat([mid_dists, jt.Var([sample_dist]).expand(mid_dists[..., :1].shape)], -1)

            cdf = jt.sigmoid(inv_s * sdf)
            e = inv_s * (1 - cdf) * (-iter_cos) * mid_dists.reshape(-1, 1)
            alpha = (1 - jt.exp(-e)).reshape(batch_size, n_samples).safe_clip(0.0, 1.0)
        else:
            # Original method in NeuS
            estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

            prev_cdf = jt.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = jt.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf
            # cdf = c
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).safe_clip(0.0, 1.0)

        # === inner parts
        pts_inner = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_inner[..., :, None]
        dir_inner = rays_d[:, None, :].expand(pts_inner.shape)
        pts_inner = pts_inner.reshape(-1, 3)
        dir_inner = dir_inner.reshape(-1, 3)
        sdf_output_inner = self.sdf_network(pts_inner)
        sdf_inner = sdf_output_inner[:, :1]
        feature_vector_inner = sdf_output_inner[:, 1:]
        # no_normal setup used in inner color network
        # gradient_inner = self.sdf_network.gradient(pts_inner).squeeze(1)
        # normal_inner = gradient_inner / (gradient_inner.norm(dim=-1, keepdim=True) + 1e-7)
        sampled_color_inner = self.color_network_inner(pts_inner, None, dir_inner, feature_vector_inner)
        sampled_color_inner = sampled_color_inner.reshape(batch_size, self.n_inner_samples, 3)

        sdf_inner = sdf_inner.reshape((batch_size, self.n_inner_samples))
        beta = self.beta_network.abs()
        sigma_inner = self.factor * laplace(sdf_inner, beta)
        dists = z_vals_inner[..., 1:] - z_vals_inner[..., :-1]
        dists = jt.concat([dists, jt.Var([sample_dist]).expand(dists[..., :1].shape)], -1)
        alpha_inner = self.raw2alpha(sigma_inner, dists)

        # ========
        pts_norm = jt.norm(pts, p=2, dim=-1, keepdim=True, eps=1e-6).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = jt.concat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + \
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = jt.concat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha + 1e-6], -1), -1)[:, :-1]
        weights_inner = alpha_inner * jt.cumprod(jt.concat([jt.ones([batch_size, 1]), 1. - alpha_inner + 1e-6], -1),
                                                 -1)[:, :-1]
        weights_inner[:, :-1] = 0.0
        if self.inner_weight_norm:
            weights_inner = weights_inner.reshape(batch_size, self.n_inner_samples) / (
                    weights_inner.reshape(batch_size, self.n_inner_samples).sum(dim=1, keepdims=True) + 1e-7)

        color_neus = (sampled_color * weights[:, :, None]).sum(dim=1)
        color_inner = (sampled_color_inner * weights_inner[:, :, None]).sum(dim=1)

        weights_sum = weights.sum(dim=-1, keepdims=True)
        color = (1 - self.w_trans) * color_neus + self.w_trans * color_inner

        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # === Eikonal loss
        gradient_error = (jt.norm(gradients.reshape(batch_size, n_samples, 3), p=2,
                                  dim=-1, eps=1e-6) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        # ================
        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            'mid_z_vals': mid_z_vals,
            'weights': weights[:, :n_samples],
            'weights_sum': weights_sum,
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'color_surf': color_neus,
            'color_inner': color_inner,
            'beta': beta,
            'factor': self.factor,
            # 'sigma_inner': laplace(sdf, beta),
            'laplace_weights': weights_inner,
            'inner_z_vals': z_vals_inner
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples  # Assuming the region of interest is a unit sphere
        z_vals = jt.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = jt.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (jt.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = jt.concat([mids, z_vals_outside[..., -1:]], -1)
                lower = jt.concat([z_vals_outside[..., :1], mids], -1)
                t_rand = jt.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / jt.flip(z_vals_outside, dim=-1) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None
        z_vals_inner = z_vals

        # Up sample
        if self.n_importance > 0:
            with jt.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                # enlarge sampling region for 2-points
                near_low = find_near_intersection(sdf, z_vals, near) - 2 * (2 / self.n_inner_samples)
                far_high = find_far_intersection(sdf, z_vals, far) + 2 * (2 / self.n_inner_samples)
                z_vals_inner = near_low + (far_high - near_low) * jt.linspace(0.0, 1.0, self.n_inner_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2 ** i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = jt.concat([z_vals, z_vals_outside], dim=-1)
            _, z_vals_feed = jt.argsort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    z_vals_inner=z_vals_inner,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        gradients = ret_fine['gradients']

        n_samples = self.n_samples + self.n_importance
        normals = gradients * weights[:, :n_samples, None]
        if ret_fine['inside_sphere'] is not None:
            normals = normals * ret_fine['inside_sphere'][..., None]

        return {
            'color_fine': color_fine,
            'weight_sum': ret_fine['weights_sum'],
            'weight_max': jt.max(weights, dim=-1, keepdims=True),
            'gradients': gradients,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'normals': normals,
            'color_inner': ret_fine['color_inner'],
            'beta': ret_fine['beta'],
            'factor': ret_fine['factor'],
            # 'color_ref': ret_fine['color_ref'],
            'color_surf': ret_fine['color_surf'],
            'neus_weights': weights,
            'laplace_weights': ret_fine['laplace_weights'],
            'inner_z_vals': ret_fine['inner_z_vals'],
            'z_vals': z_vals,
            'sdf': ret_fine['sdf'].reshape(batch_size, n_samples),
            # 'sigma_inner': ret_fine["sigma_inner"].reshape(batch_size, n_samples)

        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))

    def __str__(self):
        return "NeuralTOReconRenderer"
