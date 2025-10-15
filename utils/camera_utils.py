import torch
import numpy as np
from numpy.linalg import inv
import cv2
import json



def compute_intrinsics_for_render_after_resize_and_crop(K, cfg):
    width_scale = cfg.img_w_model_resize / cfg.img_width_orig
    height_scale = cfg.img_h_model_resize / cfg.img_height_orig
    fx = K[0,0] * width_scale
    fy = K[1,1] * height_scale
    cx = (K[0,2] * width_scale) - ((cfg.img_w_model_resize - cfg.img_w_model_resize_and_crop) / 2)
    cy = (K[1,2] * height_scale) - ((cfg.img_h_model_resize - cfg.img_h_model_resize_and_crop) / 2)

    focal_length = torch.tensor([fx, fy], dtype=torch.float32)
    principal_point = torch.tensor([cx, cy], dtype=torch.float32)
    img_size = [cfg.img_h_model_resize_and_crop, cfg.img_w_model_resize_and_crop]

    return focal_length, principal_point, img_size

def project_points_on_image(points_3d, K, R=None, t=None):

    """ 
    Args:
        points_3d: (bs, N, 3)
        K: (bs, 3, 3)
        R: (bs, 3, 3)
        t: (bs, 3)
    Returns:
        pts_2d: (bs, N, 2)
    """
    if R is not None and t is not None:
        pts_3d_cam = R @ points_3d.transpose(-1, -2) + t.unsqueeze(-1) # (bs, 3, N)
    else:
        pts_3d_cam = points_3d.transpose(-1, -2) # (bs, 3, N)
    pts_3d_proj = (K @ pts_3d_cam).transpose(-1, -2) # (bs, N, 3)
    pts_2d = (pts_3d_proj / pts_3d_proj[..., -1].unsqueeze(-1))[..., :2] # (bs, N, 3) / (bs, N, 1) -> (bs, N, 2)
    return pts_2d


def intrisic_matrix_from_parameters(fx_fy, cx_cy):
    """
    Args:
        fx_fy: (N, 2) or (2,)
        cx_cy: (N, 2) or (2,)
    Returns:
        K: (N, 3, 3) or (3, 3)
    """
    if fx_fy.dim() == 2:
        K = torch.zeros(fx_fy.shape[0], 3, 3, dtype=torch.float32, device=fx_fy.device)
        K[:, 0, 0] = fx_fy[:,0]
        K[:, 1, 1] = fx_fy[:,1]
        K[:, 0, 2] = cx_cy[:,0]
        K[:, 1, 2] = cx_cy[:,1]
        K[:, 2, 2] = 1.
    else:
        K = torch.zeros(3, 3, dtype=torch.float32, device=fx_fy.device)
        K[0, 0] = fx_fy[0]
        K[1, 1] = fx_fy[1]
        K[0, 2] = cx_cy[0]
        K[1, 2] = cx_cy[1]
        K[2, 2] = 1.
    return K

def get_camera_intrinsics(camera_file):
    with open(camera_file) as f:
        data = json.load(f)
        cam_intrinsics = data['camera_settings'][0]['intrinsic_settings']
        K = np.array([[cam_intrinsics['fx'],      0.,     cam_intrinsics['cx']],
                      [0,     cam_intrinsics['fy'],       cam_intrinsics['cy']],
                      [0.,    0.,     1.]])
        return torch.from_numpy(K).float()


def get_K_crop_resize(K, boxes, orig_size, crop_resize):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4, )
    K = K.float()
    boxes = boxes.float()
    new_K = K.clone()

    orig_size = torch.tensor(orig_size, dtype=torch.float)
    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci # (3 - 1) - (1 + 3)

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K


def project_3d_points_to_img_plane(points_3d, T, K):
    assert K.shape[0] == 3
    if T.shape[0] == 3:
        T = np.vstack((T, np.array([[0., 0., 0., 1.]])))
    
    if points_3d.shape[1] == 3:
        # points are expected in (d, N) shape
        points_3d = np.vstack((points_3d.transpose(-1, -2), np.ones(points_3d.shape[0])))

    mask = np.eye(3,4)
    points_3d_wrt_cam =  mask @ T @ points_3d
    points_3d_wrt_cam = (1 / points_3d_wrt_cam[-1, :]) * points_3d_wrt_cam
    points_2d = K @ points_3d_wrt_cam
    points_2d = points_2d[:-1, :].transpose(-1, -2)
    return points_2d
