import numpy as np

def normalize_camera_poses(poses: np.ndarray, target_radius=1.0):
    """
    Args:
        target_radius: result radius value
        poses: all camera poses, make sure these poses are Camera-to-World transformation matrix
    Returns:
        normalized camera poses, if the views are under hemisphere or sphere,
        the normalized camera poses will lay inside the target_radius (unit) sphere
    """
    if poses.shape[1:] != (4, 4):
        raise ValueError("each camera pose must be 4x4 matrix!")
    # normalize camera_pose according to code in NeRF++,
    # links: https://github.com/Kai-46/nerfplusplus/blob/master/colmap_runner/normalize_cam_dict.py
    camera_centers = []
    n_camera = poses.shape[0]
    for i in range(n_camera):
        camera_centers.append(poses[i, :3, 3:4])  # extract tx, ty, tz, shape: 3x1
    camera_centers = np.hstack(camera_centers)  # 3xn
    avg_camera_center = np.mean(camera_centers, axis=1, keepdims=True)  # center of tx, ty and tz in all camera
    dist = np.linalg.norm(camera_centers - avg_camera_center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    avg_camera_center = avg_camera_center.flatten()
    radius = diagonal * 1.1
    translate = -avg_camera_center
    scale = target_radius / radius
    translate = translate[:, np.newaxis]
    translate = np.repeat(translate, n_camera, axis=1)
    # translate = np.zeros_like(translate)
    camera_centers = (camera_centers + translate) * scale
    normalized_pose = poses.copy()
    normalized_pose[:, :3, 3] = camera_centers.T
    return normalized_pose, translate, scale

def normalize_camera_poses_jt(poses, target_radius):
    poses_np = poses.cpu().numpy()
    normalized_pose, translate, scale = normalize_camera_poses(poses_np, target_radius)
    # we only need scale to compute intersection points
    return scale
