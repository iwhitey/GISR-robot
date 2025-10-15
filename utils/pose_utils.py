import torch
import cv2 as cv
import numpy as np
from kornia.geometry.conversions import convert_points_to_homogeneous
from pytorch3d.ops.perspective_n_points import efficient_pnp
from pytorch3d.renderer import look_at_view_transform

from dataclasses import dataclass

import warnings



def pose_from_pnp(pts_2d, q, K, robot):
    """
    Computes 6DoF pose from joint angles via FK, 2d keypoints and PnP.
    Args:
        pts_2d: (N, 2)
        q: (N, d)
        K: (N, 3, 3)
        robot: robot object 
    Returns:
        R: (N, 3, 3)
        t: (N, 3)
    """
    _, t_robot = robot.get_T_all_links(q)
    pts_3d_obj = t_robot[:, [1,2,3,4,5,6,7]] # for panda robot
    R, t = solve_pnp_gpu(pts_2d, pts_3d_obj, K)
    return R, t, pts_3d_obj



def solve_pnp_gpu(pts_2d, pts_3d_obj, K):
    """ 
    Computes 6DoF pose from 2D-3D correspondences via PnP on GPU. Rotation is transposed to correspond to DREAM_Real_poses definition (row-wise), i.e. opencv-PnP and pytorch3d-PnP return mutually inverse rotatations since pytroch3d uses row-wise transforms. 
    Args:
        pts_2d: (N, 2)
        pts_3d_obj: (N, 3)
        K: (N, 3, 3)
    Returns:
        R: (N, 3, 3)
        t: (N, 3)
    """

    pts_2d_hom = convert_points_to_homogeneous(pts_2d)
    pts_2d_uncalibrated_hom = torch.linalg.solve(K, pts_2d_hom.transpose(-1, -2)) # faster and more stable than direct inverse
    # pts_2d_uncalibrated_hom = K.inverse() @ pts_2d_hom.transpose(-1, -2) # K.float().inverse() @
    pts_2d_uncalibrated = pts_2d_uncalibrated_hom.transpose(-1, -2)[:, :, :2]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        pnp_solution = efficient_pnp(pts_3d_obj, pts_2d_uncalibrated)
    R = pnp_solution.R.transpose(-1, -2)
    t = pnp_solution.T
    return R, t

def random_look_at_pose(size, interval = {"dist": [-0.2, 0.2], "elev": [-5, 5], "azim": [-5, 5]}, at = ((0, 0, 0),), device='cuda'):
    """
    Generates random look-at pose.
    Returns:
        R: (size, 3, 3)
        t: (size, 3
    """
    all_are_lists = all(isinstance(interval[key], list) for key in ["dist", "elev", "azim"])
    if all_are_lists:
        dist = torch.rand(size) * (interval["dist"][1] - interval["dist"][0]) + interval["dist"][0]
        elev = torch.rand(size) * (interval["elev"][1] - interval["elev"][0]) + interval["elev"][0]
        azim = torch.rand(size) * (interval["azim"][1] - interval["azim"][0]) + interval["azim"][0]
    else:
        dist = torch.tensor(interval["dist"], device=device)
        elev = torch.tensor(interval["elev"], device=device)
        azim = torch.tensor(interval["azim"], device=device)
 
    _at = size * at
    R, t = look_at_view_transform(dist, elev, azim, at=_at, device=device)
    return R, t


def compute_rotation_matrix_from_ortho6d(poses):
    """
    Code from https://github.com/papagina/RotationContinuity
    On the Continuity of Rotation Representations in Neural Networks
    Zhou et al. CVPR19
    https://zhouyisjtu.github.io/project_rotation/rotation.html
    """
    assert poses.shape[-1] == 6
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, p=2, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    matrix = torch.stack((x, y, z), -1)
    return matrix
 

def solve_pnp(pts2d, pts3d, K, ini_pose=None):
    bs = pts2d.size(0)
    n = pts2d.size(1)
    device = pts2d.device
    K_np = np.array(K.detach().cpu())
    P_6d = torch.zeros(bs,6,device=device)

    for i in range(bs):
        pts2d_i_np = np.ascontiguousarray(pts2d[i].detach().cpu()).reshape((n,1,2))
        pts3d_i_np = np.ascontiguousarray(pts3d[i].detach().cpu()).reshape((n,3))
        if ini_pose is None:
            #_, rvec0, T0, _ = cv.solvePnPRansac(objectPoints=pts3d_i_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, confidence=0.9999 ,reprojectionError=1)
            _, rvec0, T0 = cv.solvePnP(objectPoints=pts3d_i_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_EPNP)
        else:
            rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
            T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
        _, rvec, T = cv.solvePnP(objectPoints=pts3d_i_np, imagePoints=pts2d_i_np, cameraMatrix=K_np, distCoeffs=None, flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True, rvec=rvec0, tvec=T0)
        angle_axis = torch.tensor(rvec,device=device,dtype=torch.float).view(1, 3)
        T = torch.tensor(T,device=device,dtype=torch.float).view(1, 3)
        P_6d[i,:] = torch.cat((angle_axis,T),dim=-1)

    return P_6d


@dataclass(frozen=True)
class UR5_Rotation:

    rot_x = torch.Tensor([[1, 0, 0],
                            [0, 0, 1],
                            [0, -1, 0]]).unsqueeze(0).float()
    
    rot_z = torch.Tensor([[-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 1]]).unsqueeze(0).float()
    rot_xz = rot_x @ rot_z


@dataclass(frozen=True)
class DREAM_Real_Poses:
    """Rotation matrices are col-wise, i.e. as computed by opencv-PnP and need to be transposed for pytorch3d rendering (row-wise)"""

    kinect360 = torch.Tensor([[-0.8425, 0.5357, -0.0565, 0.3212],
                                [0.1939, 0.2037, -0.9596, 0.3877],
                                [-0.5026, -0.8195, -0.2755, 1.3339]]).float()

    realsense = torch.Tensor([[-0.8247,  0.5630, -0.0545,  0.1979],
                                [ 0.1600,  0.1397, -0.9772,  0.4338],
                                [-0.5425, -0.8146, -0.2052,  1.3987]]).float()
    
    azure = torch.Tensor([[ 0.8470,  0.5315,  0.0026, -0.1975],
                            [ 0.1016, -0.1571, -0.9823,  0.3972],
                            [-0.5217,  0.8323, -0.1871,  1.0454]]).float()


