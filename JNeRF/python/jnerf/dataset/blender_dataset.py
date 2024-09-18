import jittor as jt

import cv2 as cv
import numpy as np
import os
import json
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from jnerf.utils.registry import DATASETS
from jnerf.utils.bbox_utils import get_bbox_from_mask


@DATASETS.register_module()
class BlenderDataset:
    def __init__(self, dataset_dir, d_type="train"):
        super(BlenderDataset, self).__init__()
        print('Load data: Begin')
        self.d_type = d_type

        self.data_dir = dataset_dir
        self.camera_outside_sphere = True  # conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = 1.1  # conf.get_float('scale_mat_scale', default=1.1)
        self.images_lis = sorted(glob(os.path.join(self.data_dir, "{}/*.png".format(d_type))))
        # print(os.path.join(self.data_dir, "{}/*.png".format(d_type)))

        self.n_images = len(self.images_lis)
        imlist = [cv.imread(im_name) for im_name in self.images_lis]
        for i in range(len(imlist)):
            if len(imlist[i].shape) == 4:
                imlist[i] = cv.cvtColor(imlist[i], cv.COLOR_BGRA2BGR)
            imlist[i] = jt.array(imlist[i])

        self.images = jt.stack(imlist) / 255.0

        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks = jt.stack([jt.array(cv.imread(im_name)) for im_name in self.masks_lis]) / 256.0

        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        # print(self.n_images)
        # ================= read pose data
        self.intrinsics_all = []
        self.pose_all = []
        flip_yz = np.eye(4)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        camera_dict = json.load(open(os.path.join(self.data_dir, "transforms_{}.json".format(d_type))))
        camera_angle_x = float(camera_dict['camera_angle_x'])
        frames = sorted(camera_dict["frames"], key=lambda x: x["file_path"])
        focal = .5 * self.W / np.tan(.5 * camera_angle_x)
        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_images)]
        K = np.array([
            [focal, 0, self.W // 2, 0],
            [0, focal, self.H // 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)
        for idx, frame in enumerate(frames):
            C2W = np.array(frame["transform_matrix"])
            C2W = np.matmul(C2W, flip_yz)
            C2W = C2W.astype(np.float32)
            self.intrinsics_all.append(jt.Var(K))
            self.pose_all.append(jt.Var(C2W).float())

        self.intrinsics_all = jt.stack(self.intrinsics_all)  # [n_images, 4, 4]
        self.intrinsics_all_inv = jt.linalg.inv(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = jt.stack(self.pose_all)
        self.pose_origin = self.pose_all.clone()

        self.world_scale = self.scale_mats_np[0][0, 0]
        self.world_trans = 0.0

        object_bbox_min = np.array([-1.21, -1.21, -1.21, 1.0])
        object_bbox_max = np.array([1.21, 1.21, 1.21, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.eye(4).astype(np.float32)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def __len__(self):
        return self.n_images

    def jt_matmul(self, a, b):

        h, w, _, _ = b.shape
        a = a.expand(h, w, 1, 1)

        return jt.matmul(a, b)

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = self.jt_matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None])  # W, H, 3
        p = p.squeeze(dim=3)
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True, eps=1e-6)  # W, H, 3
        rays_v = self.jt_matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None])  # W, H, 3
        rays_v = rays_v.squeeze(dim=3)
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = jt.randint(low=0, high=self.W, shape=[batch_size])
        pixels_y = jt.randint(low=0, high=self.H, shape=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        point = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        bs = point.shape[0]
        point = jt.matmul(self.intrinsics_all_inv[img_idx, :3, :3].expand(bs, 1, 1), point[:, :, None])  # batch_size, 3
        point = point.squeeze(dim=2)
        rays_v = point / jt.norm(point, p=2, dim=-1, keepdim=True, eps=1e-6)  # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, :3, :3].expand(bs, 1, 1), rays_v[:, :, None])  # batch_size, 3
        rays_v = rays_v.squeeze(dim=2)
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        # print(rays_o.shape)
        # print(color.shape)
        # print(mask.shape)
        return jt.concat([rays_o, rays_v, color, mask[:, :1]], dim=-1)  # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = jt.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / jt.norm(p, p=2, dim=-1, keepdim=True, eps=1e-6)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = jt.Var(pose[:3, :3])
        trans = jt.Var(pose[:3, 3])
        rays_v = jt.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = jt.sum(rays_d ** 2, dim=-1, keepdims=True)
        b = 2.0 * jt.sum(rays_o * rays_d, dim=-1, keepdims=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        if len(img.shape) == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def gen_ray_at_pixel(self, img_idx, pixel, resolution_level=1):
        pass

@DATASETS.register_module()
class BlenderDatasetStage2(BlenderDataset):
    def __init__(self, dataset_dir, d_type="train"):
        super().__init__(dataset_dir, d_type)

        # self.images_np = self.images_np[..., ::-1].astype(np.float32)
        # self.images = jt.array(self.images_np)

        # does not use normalize
        # _, __, self.r_radius = normalize_camera_poses_jt(self.pose_origin, target_radius=1)
        self.r_radius = 1
        # self.r_radius = 1 / (self.r_radius.numpy()[0])
        print(self.r_radius)

        # pre-compute rgb_grad
        # self.rgb_grads = []
        # for rgb_img_idx in range(self.n_images):
        #     rgb_img = self.images_np[rgb_img_idx, ...]
        #     rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)  # [H, W]
        #     rgb_grad_x = cv2.Sobel(rgb_gray, cv2.CV_32F, 1, 0, ksize=cv2.FILTER_SCHARR)  # [H, W]
        #     rgb_grad_y = cv2.Sobel(rgb_gray, cv2.CV_32F, 0, 1, ksize=cv2.FILTER_SCHARR)  # [H, W]
        #     rgb_grad = cv2.magnitude(rgb_grad_x, rgb_grad_y)  # [H, W]
        #     self.rgb_grads.append(rgb_grad)
        # self.rgb_grads = np.stack(self.rgb_grads)
        # self.rgb_grads = jt.from_numpy(self.rgb_grads).float().unsqueeze(-1)
    
    def gen_rays_at(self, img_idx, factor=1):
        l = int(1 / factor)
        tx = jt.linspace(0, self.W - 1, self.W // l)
        ty = jt.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = jt.meshgrid(tx, ty)
        p = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = self.jt_matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze(dim=3)  # W, H, 3
        rays_v_norm = jt.norm(p, dim=-1, keepdim=True)
        rays_v = p / rays_v_norm  # W, H, 3
        rays_v = self.jt_matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze(dim=3)  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return jt.concat([rays_o.transpose(0, 1), rays_v.transpose(0, 1), rays_v_norm.transpose(0, 1)], dim=-1)

    def get_radii(self, img_idx):
        C2W = self.pose_all[img_idx, ...]
        K_inv = self.intrinsics_all_inv[img_idx, ...]
        tx = jt.linspace(0, self.W - 1, self.W)
        ty = jt.linspace(0, self.H - 1, self.H)
        u, v = jt.meshgrid(tx, ty)
        uv = jt.stack([u, v], dim=-1) + 0.5
        dots_sh = list(uv.shape[:-1])
        uv = uv.reshape(-1, 2)
        uv = jt.concat((uv, jt.ones_like(uv[..., 0:1])), dim=-1)
        ray_d = jt.matmul(
            jt.matmul(uv, K_inv[:3, :3].transpose(1, 0)),
            C2W[:3, :3].transpose(1, 0),
        ).reshape(
            dots_sh
            + [
                3,
            ]
        )
        dx = jt.sqrt(jt.sum((ray_d[:-1, :, :] - ray_d[1:, :, :]) ** 2, dim=-1))
        dx = jt.concat([dx, dx[-2:-1, :]], dim=0)  # [H, W]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx * 2 / np.sqrt(12)
        return jt.max(radii)

    def image_at(self, idx, factor):
        resolution_level = int(1 / factor)
        img = cv.imread(self.images_lis[idx])
        if len(img.shape) == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img  = (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
        return img / 255.0

    def mask_at(self, idx, factor):
        resolution_level = int(1 / factor)
        mask = cv.imread(self.masks_lis[idx])
        mask = mask / 255.0
        mask = cv.resize(mask, (self.W // resolution_level, self.H // resolution_level))
        mask = mask > 0.5
        return jt.array(mask)

    def gen_masked_random_rays_at(self, img_idx, batch_size):
        bbox_top, bbox_left, bbox_bottom, bbox_right = get_bbox_from_mask(self.masks[img_idx], enlarge=10)
        pixels_x = jt.randint(low=bbox_left, high=bbox_right+1, shape=[batch_size])
        pixels_y = jt.randint(low=bbox_top, high=bbox_bottom+1, shape=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]  # batch_size, 3
        point = jt.stack([pixels_x, pixels_y, jt.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        bs = point.shape[0]
        point = jt.matmul(self.intrinsics_all_inv[img_idx, :3, :3].expand(bs, 1, 1), point[:, :, None])  # batch_size, 3
        point = point.squeeze(dim=2)
        rays_v = point / jt.norm(point, p=2, dim=-1, keepdim=True, eps=1e-6)  # batch_size, 3
        rays_v = jt.matmul(self.pose_all[img_idx, :3, :3].expand(bs, 1, 1), rays_v[:, :, None])  # batch_size, 3
        rays_v = rays_v.squeeze(dim=2)
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)  # batch_size, 3
        return jt.concat([rays_o, rays_v, color, mask[:, :1]], dim=-1)  # batch_size, 10