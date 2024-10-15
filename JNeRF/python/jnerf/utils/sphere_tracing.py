import jittor as jt
from jittor import nn

class SphereTracer(nn.Module):
    """Implementing Sphere Tracing using Jittor"""

    def __init__(self, sdf_threshold=5e-5,
                 sphere_tracing_iters=32,
                 n_steps=128,
                 max_num_rays=200000):
        # sdf values of convergent points must be inside [-sdf_threshold, sdf_threshold]
        super().__init__()
        self.coarse_sdf_threshold = sdf_threshold * 4
        self.sdf_threshold = sdf_threshold
        # sphere tracing hyper-params
        self.sphere_tracing_iters = sphere_tracing_iters
        # dense sampling hyper-params
        self.n_steps = n_steps
        self.max_num_pts = max_num_rays
        self.max_num_rays = max_num_rays
    
    @jt.no_grad()
    def execute(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask, ray_d_norm=None):
        """
        Ray Tracing given start point at ray_o and view at ray_d
        Args:
            sdf: SDF Network
            ray_o: tracer start at `ray_o` points
            ray_d: ray direction for trace
            min_dis: min distance for searching, can simply set to 0
            max_dis: max distance for searching, can simply set to 1 if normalization applied in camera pose
            work_mask: object mask
            ray_d_norm: scale corresponding to unit vector
        Returns:
            result: {}
        """
        merge_results = None
        if ray_d_norm is None:
            ray_d_norm = jt.ones_like(work_mask)
        for ray_o_split, ray_d_split, ray_d_norm_split, mask_split, min_dis_split, max_dis_split in zip(
                jt.split(ray_o, self.max_num_rays, dim=0),
                jt.split(ray_d, self.max_num_rays, dim=0),
                jt.split(ray_d_norm, self.max_num_rays, dim=0),
                jt.split(work_mask, self.max_num_rays, dim=0),
                jt.split(min_dis, self.max_num_rays, dim=0),
                jt.split(max_dis, self.max_num_rays, dim=0)
        ):
            results = self.ray_trace(
                sdf,
                ray_o_split,
                ray_d_split,
                min_dis_split,
                max_dis_split,
                mask_split,
                ray_d_norm_split,
            )
            if merge_results is None:
                merge_results = dict(
                    [
                        (
                            x,
                            [
                                results[x],
                            ],
                        )
                        for x in results.keys()
                        if isinstance(results[x], jt.Var)
                    ]
                )
            else:
                for x in results.keys():
                    merge_results[x].append(results[x])

        for x in list(merge_results.keys()):
            results = jt.concat(merge_results[x], dim=0)
            merge_results[x] = results

        return merge_results
    
    def ray_trace(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask, scale):
        (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_start_sdf,
            acc_start_dis,
        ) = self.sphere_tracing(sdf, ray_o, ray_d, min_dis, max_dis, work_mask)
        sampler_work_mask = unfinished_mask_start

        if sampler_work_mask.sum() > 0:
            tmp_mask = (curr_start_sdf[sampler_work_mask] > 0.0).float()
            sampler_min_dis = (
                    tmp_mask * acc_start_dis[sampler_work_mask] + (1.0 - tmp_mask) * min_dis[sampler_work_mask]
            )
            sampler_max_dis = (
                    tmp_mask * max_dis[sampler_work_mask] + (1.0 - tmp_mask) * acc_start_dis[sampler_work_mask]
            )

            (sampler_convergent_mask, sampler_points, sampler_sdf, sampler_dis,) = self.ray_sampler(
                sdf,
                ray_o[sampler_work_mask],
                ray_d[sampler_work_mask],
                sampler_min_dis,
                sampler_max_dis,
            )
            convergent_mask[sampler_work_mask] = sampler_convergent_mask
            curr_start_points[sampler_work_mask] = sampler_points
            curr_start_sdf[sampler_work_mask] = sampler_sdf
            acc_start_dis[sampler_work_mask] = sampler_dis

        ret_dict = {
            "convergent_mask": convergent_mask,
            "points": curr_start_points,
            "sdf": curr_start_sdf,
            "distance": acc_start_dis,
        }
        return ret_dict

    def sphere_tracing(self, sdf, ray_o, ray_d, min_dis, max_dis, work_mask):
        """Run sphere tracing algorithm for max iterations"""
        iters = 0
        unfinished_mask_start = work_mask.clone()
        acc_start_dis = min_dis.clone()
        curr_start_points = ray_o + ray_d * acc_start_dis.unsqueeze(-1)
        curr_sdf_start = sdf(curr_start_points)
        # print(curr_start_points)
        while True:
            # Check convergence
            unfinished_mask_start = (
                    unfinished_mask_start & (curr_sdf_start.abs() > self.sdf_threshold) & (acc_start_dis < max_dis)
            )
            if iters == self.sphere_tracing_iters or unfinished_mask_start.sum() == 0:
                break
            iters += 1

            # Make step
            tmp = curr_sdf_start[unfinished_mask_start]
            acc_start_dis[unfinished_mask_start] += tmp
            curr_start_points[unfinished_mask_start] += ray_d[unfinished_mask_start] * tmp.unsqueeze(-1)
            curr_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        convergent_mask = (
                work_mask
                & jt.logical_not(unfinished_mask_start)
                & (curr_sdf_start.abs() <= self.sdf_threshold)
                & (acc_start_dis < max_dis)
        )
        return (
            convergent_mask,
            unfinished_mask_start,
            curr_start_points,
            curr_sdf_start,
            acc_start_dis,
        )
    
    def ray_sampler(self, sdf, ray_o, ray_d, min_dis, max_dis, n_steps=None):
        if n_steps is None:
            n_steps = self.n_steps
        """Sample the ray in a given range and perform rootfinding on ray segments which have sign transition"""
        intervals_dis = (
            jt.linspace(0, 1, steps=n_steps).float().reshape(1, self.n_steps)
        )  # [1, n_steps]
        intervals_dis = min_dis.unsqueeze(-1) + intervals_dis * (
                max_dis.unsqueeze(-1) - min_dis.unsqueeze(-1)
        )  # [n_valid, n_steps]
        points = ray_o.unsqueeze(-2) + ray_d.unsqueeze(-2) * intervals_dis.unsqueeze(-1)  # [n_valid, n_steps, 3]
        sdf_val = []
        for pnts in jt.split(points.reshape(-1, 3), self.max_num_rays, dim=0):
            sdf_val.append(sdf(pnts))
        sdf_val = jt.concat(sdf_val, dim=0).reshape(-1, n_steps)
        sampler_pts = jt.zeros_like(ray_d)
        sampler_sdf = jt.zeros_like(min_dis)
        sampler_dis = jt.zeros_like(min_dis)
        tmp = jt.nn.sign(sdf_val) * jt.arange(n_steps, 0, -1).reshape(
            1, n_steps
        )
        min_val, min_idx = jt.arg_reduce(tmp, 'min', dim=-1, keepdims=False)
        rootfind_work_mask = (min_val < 0.0) & (min_idx >= 1)
        n_rootfind = rootfind_work_mask.sum()
        if n_rootfind > 0:
            min_idx = min_idx[rootfind_work_mask].unsqueeze(-1)
            z_low = jt.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(-1)
            sdf_low = jt.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx - 1).squeeze(-1)
            z_high = jt.gather(intervals_dis[rootfind_work_mask], dim=-1, index=min_idx).squeeze(-1)
            sdf_high = jt.gather(sdf_val[rootfind_work_mask], dim=-1, index=min_idx).squeeze(-1)

            p_pred, z_pred, sdf_pred = self.rootfind(
                sdf,
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                ray_o[rootfind_work_mask],
                ray_d[rootfind_work_mask],
            )

            sampler_pts[rootfind_work_mask] = p_pred
            sampler_sdf[rootfind_work_mask] = sdf_pred
            sampler_dis[rootfind_work_mask] = z_pred

        return rootfind_work_mask, sampler_pts, sampler_sdf, sampler_dis
    
    def rootfind(self, sdf, f_low, f_high, d_low, d_high, ray_o, ray_d, sdf_threshold=None):
        if sdf_threshold is None:
            sdf_threshold = self.coarse_sdf_threshold
        work_mask = (f_low > 0) & (f_high < 0)
        d_mid = (d_low + d_high) / 2.0
        i = 0
        while work_mask.any():
            p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
            f_mid = sdf(p_mid)
            ind_low = f_mid > 0
            ind_high = f_mid <= 0
            if ind_low.sum() > 0:
                d_low[ind_low] = d_mid[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if ind_high.sum() > 0:
                d_high[ind_high] = d_mid[ind_high]
                f_high[ind_high] = f_mid[ind_high]
            d_mid = (d_low + d_high) / 2.0
            work_mask &= (d_high - d_low) > 2 * sdf_threshold
            i += 1
        p_mid = ray_o + ray_d * d_mid.unsqueeze(-1)
        f_mid = sdf(p_mid)
        return p_mid, d_mid, f_mid
