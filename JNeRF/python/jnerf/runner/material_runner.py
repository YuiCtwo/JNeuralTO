import glob
from copy import deepcopy

import jittor as jt
import imageio
import os
import numpy as np
import tqdm

from jnerf.models.networks.neuralto_network import NeuralTOMaterial, SDFNetwork
from jnerf.utils.sphere_tracing import SphereTracer
from jittor import nn
from jnerf.models.samplers.neuralto_render import NeuralTOScatterRenderer

from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, DATASETS, OPTIMS, SAMPLERS

class CoLocatedPointLight(nn.Module):
    def __init__(self, init_light_intensity=5.0):
        super().__init__()
        self.light = jt.Var([init_light_intensity])

    def execute(self):
        return self.light


def to8b(x):
    return np.clip(x*255.0, 0.0, 255.0).astype(np.uint8)

# log10(x) = log(x) / log(10)
def log10(x):
    return jt.log(x) / jt.log(jt.array(np.array([10.])))

@jt.no_grad()
def intersect_sphere(ray_o, ray_d, r):
    """
    ray_o, ray_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    """
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -jt.sum(ray_d * ray_o, dim=-1) / jt.sum(ray_d * ray_d, dim=-1)
    # d1 = -jt.sum(ray_d * ray_o, dim=-1) / jt.sum(ray_d * ray_d, dim=-1)
    p = ray_o + d1.unsqueeze(-1) * ray_d

    tmp = r * r - jt.sum(p * p, dim=-1)
    mask_intersect = tmp > 0.0
    d2 = jt.sqrt(jt.clamp(tmp, min_v=0.0)) / jt.norm(ray_d, dim=-1)
    return mask_intersect, jt.clamp(d1 - d2, min_v=0.0), d1 + d2

def masked_l1_rgb_loss(pred, gt, mask):
    # mask_sum = mask.sum() + 1e-6
    color_error = (pred - gt) * mask
    color_fine_loss = nn.l1_loss(color_error, jt.zeros_like(color_error))
    return color_fine_loss

def mse2psnr(x): return -10. * jt.log(x) / jt.log(jt.array(np.array([10.])))
def img2mse(x, y): return jt.mean((x - y) ** 2)

def masked_psnr(pred, gt, mask):
    mse = img2mse(pred*mask, gt*mask)
    return mse2psnr(mse)

# def masked_psnr(pred, gt, mask):
#     mask_sum = mask.sum() + 1e-6
#     psnr = 20.0 * log10(1.0 / (((pred - gt) ** 2 * mask).sum() / (mask_sum * 3.0)).sqrt())
#     return psnr


class NeuralTOMaterialRunner:

    def __init__(self, is_continue=False):
        self.n_val_images = 0
        self.n_training_images = 0
        self.render_model = None
        self.material_model = None
        self.light_model = None
        self.geometry_model = None

        self.conf = get_cfg()
        self.is_training = True

        # init parameter
        # general parameters setup
        self.out_dir = self.conf.base_exp_dir
        self.save_freq = self.conf.save_freq
        self.patch_size = self.conf.batch_size
        self.num_iters = self.conf.end_iter
        self.report_freq = self.conf.report_freq
        self.val_freq = self.conf.val_freq
        self.log_freq = self.conf.log_freq
        self.use_mask = self.conf.use_mask
        # dataset parameters setup
        self.train_dataset = None
        self.val_dataset = None
        self.init_dataset()

        # ckpt parameter setup
        self.stage1_ckpt_path = self.conf.ckpt.stage1_ckpt_path
        # specify checkpoint file
        self.ckpt = self.conf.ckpt.model_ckpt_path

        # background setting
        if self.conf.background_color is not None:
            self.background_color = jt.Var([self.conf.background_color])
        else:
            self.background_color = jt.zeros([1, 3])

        self.cur_iter = 0

        self.max_num_pts = 100000  # 100x1000

        # loss parameters setup
        self.eik_weight = self.conf.loss.eik_weight
        self.roughrange_weight = self.conf.loss.roughrange_weight
        self.smoothness_weight = self.conf.loss.smoothness_weight
        # optimizer parameter
        self.lr_material = self.conf.lr.material
        self.lr_render = self.conf.lr.render
        self.lr_light = self.conf.lr.light
        self.lr_geometry = self.conf.lr.geometry

        # model parameters setup
        self.init_model()

        self.optimizer_dict = {}
        self.init_optimizer()

        # Load checkpoint
        if is_continue:
            self.load_ckpt()
        else:
            self.load_stage1_ckpt()

        self.raytracer = SphereTracer()
        self.out_dir = os.path.join(self.out_dir, str(self.patch_size))

        self.alpha0 = 0.35
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)

    @jt.no_grad()
    def raytrace_camera(self, ray_o, ray_d, ray_d_norm, mask, r=1):
        dots_sh = list(ray_o.shape[:-1])
        rays_o = ray_o.view(-1, 3)
        rays_d = ray_d.view(-1, 3)
        rays_d_norm = ray_d_norm.view(-1)
        if mask is None:
            mask = jt.ones_like(rays_o[..., 0]).bool()
        else:
            mask = mask.view(-1)
        mask_intersect, min_dis, max_dis = intersect_sphere(rays_o, rays_d, r=self.train_dataset.r_radius)
        results = self.raytracer(
            self.sdf,
            rays_o,
            rays_d,
            min_dis,
            max_dis,
            mask_intersect & mask,
            rays_d_norm
        )
        # additional processing
        for k in results.keys():
            results[k] = results[k].reshape(
                dots_sh + [-1]
            )
            if results[k].shape[-1] == 1:
                results[k] = results[k][..., 0]

        # update result
        results.update(
            {
                # "uv": uv,
                "ray_o": rays_o,
                "ray_d": rays_d,
                "ray_d_norm": rays_d_norm,
            }
        )
        return results

    def render_camera(self, rays_o, rays_d, rays_d_norm, mask=None):
        # raytrace
        results = self.raytrace_camera(rays_o, rays_d, rays_d_norm, mask)

        dots_sh = list(results["convergent_mask"].shape)
        merge_render_results = None
        for points_split, ray_d_split, ray_o_split, mask_split, ray_d_norm in zip(
                jt.split(results["points"].view(-1, 3), self.max_num_pts, dim=0),
                jt.split(results["ray_d"].view(-1, 3), self.max_num_pts, dim=0),
                jt.split(results["ray_o"].view(-1, 3), self.max_num_pts, dim=0),
                jt.split(results["convergent_mask"].view(-1), self.max_num_pts, dim=0),
                jt.split(results["ray_d_norm"].view(-1), self.max_num_pts, dim=0)
        ):
            if mask_split.any():
                points_split, ray_d_split, ray_o_split, ray_d_norm = (
                    points_split[mask_split],
                    ray_d_split[mask_split],
                    ray_o_split[mask_split],
                    ray_d_norm[mask_split]
                )
                # TODO: add get_all function for SDF model
                sdf_split, feature_split, normal_split = self.geometry_model.get_all(points_split,
                                                                                     is_training=self.is_training)
            else:
                points_split, ray_d_split, ray_o_split, normal_split, feature_split = (
                    jt.Var([]),
                    jt.Var([]),
                    jt.Var([]),
                    jt.Var([]),
                    jt.Var([]),
                )

            # with jt.set_grad_enabled(self.is_training):
            render_results = self.render_material(
                points_split, normal_split, feature_split,
                ray_d_split, ray_o_split, mask_split
            )
            # ===========================
            # prepare for result merging
            if merge_render_results is None:
                merge_render_results = dict([
                    (x, [render_results[x], ],) for x in render_results.keys()
                ])
            else:
                for x in render_results.keys():
                    merge_render_results[x].append(render_results[x])

        # merge result
        for x in list(merge_render_results.keys()):
            tmp = jt.concat(merge_render_results[x], dim=0).reshape(
                dots_sh
                + [
                    -1,
                ]
            )
            if tmp.shape[-1] == 1:
                tmp = tmp.squeeze(-1)
            merge_render_results[x] = tmp

        results.update(merge_render_results)

        return results

    def load_stage1_ckpt(self):
        print("Reloading model from stage-1 checkpoint: ", self.stage1_ckpt_path)
        ckpt = jt.load(self.stage1_ckpt_path)
        try:
            self.geometry_model.load_state_dict(ckpt["sdf_network"])
            laplace_beta = ckpt["beta"]
            laplace_factor = ckpt["alpha_factor"]
            self.render_model.laplace_factor = laplace_factor
            self.render_model.laplace_beta = laplace_beta
            print("Load laplace param: beta={}, factor={}".format(self.render_model.laplace_beta,
                                                                  self.render_model.laplace_factor))
            
            self.material_model.diffuse_albedo_network.load_state_dict(ckpt["color_network"])
            
        except Exception as e:
            print("Load failed due to: ", e)

    def load_ckpt(self):
        if self.ckpt is None:
            ckpt_dir = os.path.join(self.out_dir, "checkpoints")
            ckpt_fpaths = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pth"))
            path2step = lambda x: int(os.path.basename(x)[len("ckpt_"): -4])
            ckpt_fpaths = sorted(ckpt_fpaths, key=path2step)
            ckpt_path = ckpt_fpaths[-1]

        else:
            ckpt_path = self.ckpt
        print("Reloading model from checkpoint: ", ckpt_path)
        ckpt = jt.load(ckpt_path)
        try:
            self.geometry_model.load_state_dict(ckpt["sdf_network"])
            print("Load laplace param: beta={}, factor={}".format(self.render_model.laplace_beta,
                                                                  self.render_model.laplace_factor))
            self.material_model.load_state_dict(ckpt["material_network"])
            self.light_model.load_state_dict(ckpt["light_network"])
        except Exception as e:
            print("Load failed due to: ", e)

    def init_model(self):
        # self.material_model_param = []
        # material model
        self.material_model = build_from_cfg(self.conf.material, NETWORKS)
        # self.material_model = NeuralTOMaterial(self.conf["model.material"])
        # self.material_model_param += list(self.material_model.parameters())
        # sdf model
        self.geometry_model = SDFNetwork(**self.conf.model)
        # self.geometry_model = SDFNetwork(**self.conf["model.geometry_network.sdf_network"]).to(self.device)
        self.sdf = lambda x: self.geometry_model(x)[..., 0]
        # render model
       
        self.light_model = CoLocatedPointLight(**self.conf.light)
        
        self.render_model = NeuralTOScatterRenderer(**self.conf.render)
        # self.scattering_model = MipSSS()
        # self.render_model.network_setup(self.scattering_model)
        # self.render_model_param = list(self.render_model.parameters())

    def save_ckpt(self):
        checkpoint = {
            "iter": self.cur_iter,
            "sdf_network": self.geometry_model.state_dict(),
            "render_network": self.render_model.state_dict(),
            "material_network": self.material_model.state_dict(),
            "light_network": self.light_model.state_dict()
        }

        os.makedirs(os.path.join(self.out_dir, "checkpoints"), exist_ok=True)
        jt.save(
            checkpoint,
            os.path.join(
                self.out_dir,
                "checkpoints",
                "ckpt_{:0>6d}.pth".format(self.cur_iter),
            ),
        )

    def init_optimizer(self):
        material_optimizer = build_from_cfg(self.conf.optim.material, OPTIMS, params=self.material_model.parameters())
        sdf_optimizer = build_from_cfg(self.conf.optim.sdf, OPTIMS, params=self.geometry_model.parameters())
        render_optimizer = build_from_cfg(self.conf.optim.render, OPTIMS, params=self.render_model.parameters())
        light_optimizer = build_from_cfg(self.conf.optim.light, OPTIMS, params=self.light_model.parameters())

        self.optimizer_dict["sdf_optimizer"] = sdf_optimizer
        self.optimizer_dict["material_optimizer"] = material_optimizer
        self.optimizer_dict["render_optimizer"] = render_optimizer
        self.optimizer_dict["light_optimizer"] = light_optimizer

    def train(self):
        for it in tqdm.tqdm(range(self.num_iters)):
            self.render_model.dist = self.dist_training
            self.cur_iter = it
            # gen random idx
            idx = np.random.randint(0, self.n_training_images - 1)
            if self.use_mask:
                data = self.train_dataset.gen_masked_random_rays_at(idx, self.patch_size ** 2)
            else:
                data = self.train_dataset.gen_random_rays_at(idx, self.patch_size ** 2)
            rays_o, rays_d, rays_d_norm, true_rgb, mask = (
                data[:, :3],
                data[:, 3:6],
                data[:, 6:7],
                data[:, 7:10],
                data[:, 10:11],
            )
            gt_mask = mask.bool()
            # set radius for mipmap_r
            r = self.train_dataset.get_radii(idx)
            self.render_model.r = r

            results = self.render_camera(rays_o, rays_d, rays_d_norm)

            # loss
            img_loss = jt.Var([0.0])
            roughrange_loss = jt.Var([0.0])
            eik_loss = jt.Var([0.0])
            loss = jt.Var([0.0])

            # if not self.use_mask:
            if self.use_mask:
                gt_mask = gt_mask.squeeze(-1)
                true_rgb[~gt_mask] = 0
            # use gt mask for supervised
            mask = results["convergent_mask"].unsqueeze(-1)
            if mask.any():
                # ====================
                img_loss = masked_l1_rgb_loss(results["color"], true_rgb, jt.ones_like(mask))
                # ======================
                roughrange_loss = ((results["specular_roughness"] - self.alpha0).abs() * mask.squeeze(
                    -1)).sum() / mask.sum() * self.roughrange_weight
                # ======================
                # add additional random points
                # eik_points = jt.init.uniform((self.patch_size * self.patch_size // 4, 3), low=-1, high=1)
                # add_eik_grad = self.geometry_model.gradient(eik_points).view(-1, 3)
                # add_eik_cnt = add_eik_grad.shape[0]
                # eik_loss = (((add_eik_grad.norm(dim=-1) - 1) ** 2).sum() / add_eik_cnt) * self.eik_weight
                eik_grad = results["grad_loss"][mask.squeeze(-1)]
                eik_cnt = eik_grad.shape[0]
                eik_loss = (eik_grad.sum() / eik_cnt) * self.eik_weight
                # ========================
                # ignore smoothness loss
                loss = img_loss + roughrange_loss + eik_loss

                for one in self.optimizer_dict:
                    self.optimizer_dict[one].zero_grad()
                    self.optimizer_dict[one].backward(loss, retain_graph=True)
                    self.optimizer_dict[one].step()

            if self.cur_iter % self.save_freq == 0:
                self.save_ckpt()

            # TODO: add a report here
            # clean cache
            for x in list(results.keys()):
                del results[x]
            del loss

            if self.cur_iter % self.val_freq == 0:
                # gen random idx for val
                if self.n_val_images > 0:
                    idx = np.random.randint(0, self.n_val_images - 1)
                    # for idx in range(self.n_val_images - 1):
                    self.validate_image(idx=idx)
                else:
                    idx = np.random.randint(0, self.n_training_images-1)
                    self.validate_image(idx=idx, dataset_type="train")

    def validate_image(self, factor=0.20, idx=0, dataset_type="val"):
        self.render_model.dist = self.dist_val

        if dataset_type == "val":
            dataset = self.val_dataset
        else:
            dataset = self.train_dataset

        data = dataset.gen_rays_at(idx, factor=factor)
        # rays_o : HxWx3
        rays_o, rays_d, rays_d_norm = (
            data[..., :3],
            data[..., 3:6],
            data[..., 6:7],
        )
        gt_color_resized = dataset.image_at(idx, factor)
        r = dataset.get_radii(idx)
        self.render_model.r = r

        results = self.render_camera(rays_o, rays_d, rays_d_norm)

        for x in list(results.keys()):
            results[x] = results[x].detach().cpu().numpy()

        gt_color_im = gt_color_resized
        color_im = results["color"]
        diffuse_color_im = results["diffuse_color"]
        specular_color_im = results["specular_color"]
        normal = results["normal"]
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-10)
        normal_im = (normal + 1.0) / 2.0
        back_normal_im = (results["back_normal"] + 1.0) / 2
        diffuse_albedo_im = results["diffuse_albedo"]
        specular_albedo_im = results["specular_albedo"]

        multi_ss = results["multi_ss"]
        single_ss = results["single_ss"]
        single_color = results["single_color"]
        specular_roughness_im = np.tile(results["specular_roughness"][:, :, np.newaxis], (1, 1, 3))

        if self.is_training:
            row1 = np.concatenate([gt_color_im, normal_im, back_normal_im], axis=1)
            row2 = np.concatenate([color_im, diffuse_color_im, specular_color_im], axis=1)
            row3 = np.concatenate([multi_ss, single_ss, specular_roughness_im], axis=1)
            row4 = np.concatenate([diffuse_albedo_im, specular_albedo_im, single_color], axis=1)
            im = np.concatenate((row1, row2, row3, row4), axis=0)
            imageio.imwrite(os.path.join(self.out_dir, f"logim_{self.cur_iter}_{idx}.png"), to8b(im))
        else:
            img_name = dataset_type + str(idx)
            imageio.imwrite(os.path.join(self.out_dir, f"{img_name}.png"), to8b(color_im))
        # compute psnr and record it
        color_im_jt = jt.Var(color_im)
        gt_color_jt = jt.Var(gt_color_im).float()

        if self.use_mask:
            gt_mask_jt = dataset.mask_at(idx, factor)
        else:
            gt_mask_jt = jt.ones_like(gt_color_jt).bool()

        psnr = masked_psnr(color_im_jt, gt_color_jt, gt_mask_jt)

        print("iter-{}, img_idx-{} result: PSNR={}".format(
            self.cur_iter,
            idx,
            psnr,
        ))
        return psnr

    def validate_image_all(self):
        data_len = len(self.val_dataset)
        total_psnr = 0.0
        total_ssim = 0.0
        with open(os.path.join(self.out_dir, "metrics.txt"), "a+") as fp:
            for i in range(data_len):
                psnr = self.validate_image(factor=1, idx=i)
                total_psnr += psnr
                fp.write("img_idx:{} result: PSNR={} \n".format(
                    i,
                    psnr,
                ))
            mean_psnr = total_psnr / data_len
            print("mean result on validation dataset: PSNR={}".format(
                mean_psnr
            ))
            fp.write(
                "mean result on validation dataset: PSNR={}".format(
                    mean_psnr
                )
            )

    def init_dataset(self):
        print('Init dataset begin, from: {}'.format(self.conf.dataset["dataset_dir"]))
        dataset_cfg = deepcopy(self.conf.dataset)
        dataset_cfg["d_type"] = "train"
        self.train_dataset = build_from_cfg(dataset_cfg, DATASETS)
        dataset_cfg["d_type"] = "val"
        self.val_dataset = build_from_cfg(dataset_cfg, DATASETS)

        self.n_training_images = len(self.train_dataset)
        self.n_val_images = len(self.val_dataset)
        # if there has a normalization function, set dist_xx to the scaling factor used in normalization
        self.dist_training = 1.0
        self.dist_val = 1.0

    def render_material(self, points, ori_normals, features, rays_d, rays_o, interior_mask):
        dots_sh = list(interior_mask.shape)
        rgb = jt.zeros(
            dots_sh
            + [
                3,
            ],
        )
        diffuse_rgb = rgb.clone()
        specular_rgb = rgb.clone()
        diffuse_albedo = rgb.clone()
        specular_albedo = rgb.clone()
        specular_roughness = rgb[..., 0:1].clone()
        normals_pad = rgb.clone()
        normals_back = rgb.clone()
        multi_ss = rgb.clone()
        single_ss = rgb.clone()
        single_color = rgb.clone()

        grad_loss = rgb[..., 0:1].clone()
        roughness_grad = rgb.clone()
        # albedo_grad = rgb.clone()
        # set background
        background_color = self.background_color.repeat(dots_sh[0], 1)
        rgb = rgb + background_color

        if interior_mask.any():
            # query material
            normals = ori_normals / (ori_normals.norm(dim=-1, keepdim=True) + 1e-6)
            interior_material = self.material_model(points, normals, features)


            diffuse_albedo[interior_mask] = interior_material["diffuse_albedo"]
            specular_albedo[interior_mask] = interior_material["trans_albedo"]
            specular_roughness[interior_mask] = interior_material["specular_roughness"]
            # roughness_grad[interior_mask] = interior_material["roughness_grad"]
            # albedo_grad[interior_mask] = interior_material["albedo_grad"]

            # back raytracer
            rays_t = rays_d

            mask_intersect_split, min_dis, max_dis = intersect_sphere(points, -rays_t, r=self.train_dataset.r_radius)
            symmetry_points = points + max_dis[..., None] * rays_t
            min_dis_split = jt.zeros_like(min_dis)
            max_dis_split = jt.ones_like(max_dis)

            trace_results = self.raytracer(
                self.sdf,
                symmetry_points,
                -rays_t,
                min_dis_split,
                max_dis_split,
                mask_intersect_split,
                jt.ones_like(mask_intersect_split),
            )
            conv_mask = trace_results["convergent_mask"]
            far_p = trace_results["points"]
            sdf_split, feature_split, normal_split = self.geometry_model.get_all(far_p, is_training=self.is_training)

            back_normal = normal_split / normal_split.norm(dim=-1, keepdim=True)

            
            thickness = jt.sqrt(jt.sum(jt.pow(points - far_p, 2), dim=-1, keepdim=True))
            surf_distance = (points - rays_o).norm(dim=-1, keepdim=True)
            results = self.render_model(
                self.light_model, self.geometry_model,
                interior_material["diffuse_albedo"],
                interior_material["trans_albedo"],
                interior_material["specular_roughness"],
                points, normals, surf_distance, -rays_d,  # !!! rays_d same side with normal
                thickness=thickness, light_o=rays_o,
                conv_mask=conv_mask, rays_t=rays_t
            )

            rgb[interior_mask] = results["rgb"]
            diffuse_rgb[interior_mask] = results["diffuse_rgb"]
            specular_rgb[interior_mask] = results["specular_rgb"]

            normals_pad[interior_mask] = normals
            normals_back[interior_mask] = back_normal
            multi_ss[interior_mask] = results["multi_ss"]
            single_ss[interior_mask] = results["single_ss"]
            single_color[interior_mask] = results["single_color"]

            # grad_loss[interior_mask] = results["grad_loss"]
            # add eik-loss for first intersection
            grad_loss[interior_mask] = (ori_normals.norm(dim=-1, keepdim=True) - 1) ** 2
            # grad_loss[interior_mask] += (normal_split.norm(dim=-1, keepdim=True) - 1) ** 2

        return {
            "color": rgb,
            "diffuse_color": diffuse_rgb,
            "specular_color": specular_rgb,
            "diffuse_albedo": diffuse_albedo,
            "specular_albedo": specular_albedo,
            "specular_roughness": specular_roughness,
            "normal": normals_pad,
            "back_normal": normals_back,
            "multi_ss": multi_ss,
            "single_ss": single_ss,
            "single_color": single_color,
            "grad_loss": grad_loss,
            "roughness_grad": roughness_grad,
        }


    # interface function used in `run_net`
    # ===============
    def render(self):
        return self.validate_image_all()

    def validate_mesh(self):
        # dummy function
        print("No use for material model")
