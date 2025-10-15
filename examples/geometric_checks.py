import sys, os

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops.perspective_n_points import efficient_pnp

import kornia

import matplotlib.pyplot as plt
import cv2

from dataset.dataset import EDMDatasetReal, EDMDatasetSynthetic
from robot import PandaRobotArm
from renderer.mesh_renderer import PandaMeshRenderer
from utils.pose_utils import solve_pnp, DREAM_Real_Poses


def save_img_with_keypoints(img, pts_2d, save_path):
    img_uint8 = torchvision.transforms.functional.convert_image_dtype(img, torch.uint8)
    img_with_kp = torchvision.utils.draw_keypoints(img_uint8, pts_2d, [], colors = (240, 10, 157), radius=3, width=0)
    save_image(img_with_kp.float() / 255, save_path)



def check_projecting_fk_points_on_img_plane(kinect_dataset):
    T = DREAM_Real_Poses.kinect360.to(device) # openCV PnP pose [R.T | t] with R transposed (as pytorch3d renderer requires)
    R = T[:3, :3]; t = T[:3, 3].unsqueeze(0)
    T = torch.cat([R, t], dim=0) # pytorch3d uses row-major transform -> R transposed, t as bottom row
    K = kinect_dataset.get_cam_intrinsics()
    _, t = robot.get_T_all_links(joint_angles[0].unsqueeze(0))
    pts_3d_fk = t[0, [1,2,3,4,5,6,7]].unsqueeze(0)
    # # print(pts_3d_fk); print(pts_3d[0]) 
    pts_3d_hom = kornia.geometry.conversions.convert_points_to_homogeneous(pts_3d_fk)
    pts_3d_cam = pts_3d_hom @ T # row-major, this is what pytorch3d does, hence takes transposed R in renderer
    # pts_3d_cam = T @ pts_3d_hom.transpose(-1, -2) # col-major, requires openCV-style pose (transpose R back and t as last column)
    pts_2d_estimated = kornia.geometry.camera.perspective.project_points(pts_3d_cam, K) #pts_3d_cam.transpose(-1, -2) for col-major
    print((pts_2d[0] - pts_2d_estimated[0]).abs())
    assert 2 == 1



def check_gpu_pnp(data, robot):
    pts_2d = data['keypoints_2d_gt']
    pts_3d = data['keypoints_3d_gt']
    pose_gt = data['pose']
    _, t_robot = robot.get_T_all_links(joint_angles)
    pts_3d_obj = t_robot[:, [1,2,3,4,5,6,7]].cpu()
    K_orig = data['K']
    pts_2d_hom = kornia.geometry.conversions.convert_points_to_homogeneous(pts_2d)
    pts_2d_uncalibrated = torch.linalg.solve(K_orig, pts_2d_hom.transpose(-1, -2))
    pts_2d_uncalibrated = pts_2d_uncalibrated.transpose(-1, -2)[:, :, :2]
    pnp_solution = efficient_pnp(pts_3d_obj, pts_2d_uncalibrated)
    print("----------3D POINTS----------------")
    print(pnp_solution.x_cam - pts_3d)
    print("----------POSE----------------")
    print(pnp_solution.T)
    print(pose_gt)
    assert 1 == 2


def compute_pose_via_pnp(batch, robot, device, poses_and_proj_errors, save_path, removed_files):

    pts_2d_gt = batch['keypoints_2d_gt'].to(device)
    pts_3d_cam_gt = batch['keypoints_3d_gt'].to(device)
    K = batch['K_orig'].to(device)
    joint_angles = batch['joint_angles_gt'].to(device)
    data_id = batch['data_id'][0]
    # img = batch['img'] # 3 x H x W
    batch_size = pts_2d_gt.shape[0]

    _, t = robot.get_T_all_links(joint_angles)
    pts_3d_obj = t[:, [1,2,3,4,5,6,7]] # panda robot definition

    T = solve_pnp(pts_2d_gt, pts_3d_obj, K[0])
    R = kornia.geometry.conversions.angle_axis_to_rotation_matrix(T[:, :3])  # (n, 3, 3)
    t = T[:, 3:]   # (n, 3)
    T = torch.eye((4)).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    pts_3d_obj_hom = kornia.geometry.conversions.convert_points_to_homogeneous(pts_3d_obj)
    pts_3d_cam = T @ pts_3d_obj_hom.transpose(-1, -2)
    pts_3d_cam = pts_3d_cam[:, :3, :] # remove homogeneous coordinate
    pts_2d_estimated = kornia.geometry.camera.perspective.project_points(pts_3d_cam.transpose(-1, -2), K[0]) #pts_3d_cam.transpose(-1, -2) za col-major
    proj_2d_error = (pts_2d_gt - pts_2d_estimated).abs().mean()
    print("Proj 2d error: ", proj_2d_error)
    add_3d = (pts_3d_cam_gt - pts_3d_cam.transpose(-1, -2)).abs().mean()
    print("Proj 3d error: ", add_3d)

    save_full_path = os.path.join(save_path, data_id + "_pose.npy")
    if proj_2d_error > 0.005 or add_3d > 0.001:
         os.remove(save_full_path)
         os.remove(os.path.join(save_path, data_id + ".rgb.jpg"))
         os.remove(os.path.join(save_path, data_id + ".json"))
         removed_files.append(data_id)
   
    # np.save(save_full_path, T.cpu().numpy())
    # print(f"Saved pose to: {save_full_path}")
    poses_and_proj_errors.append((T, proj_2d_error))
    

if __name__ == "__main__":

    urdf_path = "urdfs/Panda/panda.urdf"
    ee_name = "panda_hand"
    mesh_base_dir = "."

    data_root = ""
    dataset = "panda_synth_train_dr"
    assert torch.cuda.is_available()
    device = "cuda"

    mesh_files = [mesh_base_dir + "/urdfs/Panda/meshes/visual/link0/link0.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link1/link1.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link2/link2.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link3/link3.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link4/link4.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link5/link5.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link6/link6.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/link7/link7.obj",
            mesh_base_dir + "/urdfs/Panda/meshes/visual/hand/hand.obj",
            ]
    
    cfg = argparse.ArgumentParser('').parse_args([])
    cfg.img_height_orig = 480 #240
    cfg.img_width_orig = 640 #320
    cfg.img_h_model_resize = 236
    cfg.img_w_model_resize = 236
    cfg.img_h_model_resize_and_crop = 224
    cfg.img_w_model_resize_and_crop = 224

    # train_dataset = EDMDatasetReal(data_root, dataset, complete_edm=True, robot_type="panda", cfg=cfg)
    train_dataset = EDMDatasetSynthetic(data_root, dataset, complete_edm=True, robot_type="panda", cfg=cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    robot = PandaRobotArm(urdf_path=urdf_path, ee_name=ee_name, device=device)
    
    ############ PnP poses from gt. 2D-3D correspondences ######################
    poses_and_proj_errors = []
    removed_files = []
    save_path = os.path.join(data_root, dataset)
    for idx, batch in enumerate(train_dataloader):
        print("Batch: ", idx)
        compute_pose_via_pnp(batch, robot, device, poses_and_proj_errors, save_path, removed_files)
    
    poses_and_errors_sorted = sorted(poses_and_proj_errors, key=lambda x: x[1])
    best_pose = poses_and_errors_sorted[0][0]
    print(best_pose, poses_and_errors_sorted[0][1])
    print(poses_and_errors_sorted[-1][0], poses_and_errors_sorted[-1][1])
    print(f"Removed files: {removed_files}")
    print(f"Number of removed files: {len(removed_files)}")
    sys.exit()
    ####################################################

    for data in train_dataloader:
        joint_angles = data['joint_angles_gt'].to(device)
        pts_2d = data['keypoints_2d_gt'].squeeze()
        pts_3d = data['keypoints_3d_gt'].squeeze()
        img = data['img'].squeeze()
        img_orig = data['img_orig'].squeeze()
        fx_fy_render = data['fx_fy_render'].squeeze()
        cx_cy_render = data['cx_cy_render'].squeeze()
        img_size_render = data['img_size_render'].squeeze()
        K_orig = data['K_orig'].squeeze()
        pose_gt = data['pose']
        R = pose_gt[:, :3, :3].squeeze(0)
        t = pose_gt[:, :3, 3].squeeze(0)
        robot_base_2d = data['robot_base_2d'].squeeze()
        robot_base_3d_cam = data['robot_base_3d_cam'].squeeze()

        renderer = PandaMeshRenderer(robot, mesh_files, device, 1, img_size_render)
        robot_mesh = renderer.get_batch_robot_mesh(joint_angles)

        verts = torch.cat(robot_mesh.verts_list(), dim=0)
        verts_centroid = torch.Tensor([verts[:, 0].mean(), verts[:, 1].mean(), verts[:, 2].mean()]).unsqueeze(1)
        verts_centroid_cam = R @ verts_centroid + t.unsqueeze(1) # (3, 3) @ (3, 1) + (3, 1) = (3, 1)
        verts_2d_centroid = kornia.geometry.camera.perspective.project_points(verts_centroid_cam.transpose(-1, -2), K_orig) #pts_3d_cam.transpose(-1, -2) za col-major
        print("robot mesh vertex centroid: ", verts_2d_centroid)

        _, t_robot = robot.get_T_all_links(joint_angles)
        pts_3d_obj = t_robot[:, [1,2,3,4,5,6,7]].squeeze(0).cpu()
        robot_base_3d = torch.zeros((3)).float()

        delta_p_2d = kornia.geometry.conversions.convert_points_to_homogeneous((pts_2d[1]).unsqueeze(0)) # pts_2d[1] - robot_base_2d
        delta_p_ray = torch.linalg.solve(K_orig, delta_p_2d.transpose(-1, -2))

        robot_base_2d_hom = kornia.geometry.conversions.convert_points_to_homogeneous(robot_base_2d.unsqueeze(0))
        translation_ray = torch.linalg.solve(K_orig, robot_base_2d_hom.transpose(-1, -2))
        c_sqr = ((pts_3d_obj[1] - robot_base_3d)**2).sum()
        res = ((t[-1] * delta_p_ray - t[-1] * translation_ray)**2).sum() - (c_sqr)     
        # scale = torch.sqrt(c_sqr / ((delta_p_ray - translation_ray)**2).sum())
        scale = 0.333 / ((((delta_p_ray - translation_ray)**2).sum())**0.5)
        print(f"{scale, t[-1]}")
        print(f"scale_diff = {scale - t[-1]}")
        print(res)
        print(f"Point_1_cam: {pts_3d[1]}")
        print(f"link_1_z_diff: {pts_3d[1][-1] - robot_base_3d_cam[-1]}")
        print("-------------------")

