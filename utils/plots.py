import sys, os
import numpy as np
import cv2
from robot import PandaRobotArm
from dataset.dataset import EDMDatasetReal
from torch.utils.data import DataLoader
import torch
import torchvision
import argparse
import kornia


def save_img_with_keypoints(img, pts_2d, save_path):
    img_uint8 = torchvision.transforms.functional.convert_image_dtype(img, torch.uint8)
    img_with_kp = torchvision.utils.draw_keypoints(img_uint8, pts_2d, [], colors = (240, 10, 157), radius=3, width=0)
    torchvision.utils.save_image(img_with_kp.float() / 255, save_path)


def draw_skeleton(img_path, data_id, robot, R, t, K, theta, line_width=5):

        R = R.cpu()
        t = t.cpu()
        K = K.cpu()

        _, pts_3d_robot = robot.get_T_all_links(theta)
        pts_3d_robot = pts_3d_robot[:, [0,1,2,3,4,5,6,7]].cpu()
        pts_3d_robot[:, -2, 0] += 0.0825
        pts_3d_robot[:, -1, -1] -= 0.088

        open_cv_image = cv2.imread(img_path[0])
        cv2.imwrite(f"skeletons/input_img_{data_id[0]}.jpg", open_cv_image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # pts_3d_obj_hom = kornia.geometry.conversions.convert_points_to_homogeneous(pts_3d_robot)
        # pts_3d_cam = pose @ pts_3d_obj_hom.transpose(-1, -2)
        # pts_3d_cam = pts_3d_cam[:, :3, :] # remove homogeneous coordinate
        # pts_2d = kornia.geometry.camera.perspective.project_points(pts_3d_cam.transpose(-1, -2), K) #pts_3d_cam.transpose(-1, -2) za col-major
        # pts_2d = pts_2d.int()

        pts_3d_cam = R @ pts_3d_robot.transpose(-1, -2) + t.unsqueeze(-1) # (bs, 3, N)
        pts_3d_proj = (K @ pts_3d_cam).transpose(-1, -2) # (bs, N, 3)
        pts_2d = (pts_3d_proj / pts_3d_proj[..., -1].unsqueeze(-1))[..., :2] # (bs, N, 3) / (bs, N, 1) -> (bs, N, 2)
        pts_2d = pts_2d.int()

        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,0]]), tuple([coord.item() for coord in pts_2d[0,1]]), (255, 0, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,1]]), tuple([coord.item() for coord in pts_2d[0,3]]), (0, 255, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,3]]), tuple([coord.item() for coord in pts_2d[0,4]]), (0, 0, 255), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,4]]), tuple([coord.item() for coord in pts_2d[0,5]]), (255, 255, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,5]]), tuple([coord.item() for coord in pts_2d[0,6]]), (0, 255, 255), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,6]]), tuple([coord.item() for coord in pts_2d[0,7]]), (255, 0, 255), thickness=5)
        # cv2.imshow("Image", open_cv_image)

        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"skeletons/skeleton_{data_id[0]}.jpg", open_cv_image)
        print(f"Saved skeleton_{data_id[0]}.jpg")



def test_draw_skeleton():
    data_root = ""
    dataset = ""
    cfg = argparse.ArgumentParser('').parse_args([])
    cfg.img_height_orig = 480 #240
    cfg.img_width_orig = 640 #320
    cfg.img_h_model_resize = 236
    cfg.img_w_model_resize = 236
    cfg.img_h_model_resize_and_crop = 224
    cfg.img_w_model_resize_and_crop = 224

    train_dataset = EDMDatasetReal(data_root, dataset, complete_edm=True, robot_type="panda", cfg=cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    dataloader_iter = iter(train_dataloader)

    for i in range(100):

        batch = next(dataloader_iter)

        pts_2d_gt = batch['keypoints_2d_gt']
        pts_3d_cam_gt = batch['keypoints_3d_gt']
        K = batch['K_orig']
        joint_angles = batch['joint_angles_gt']
        data_id = batch['data_id'][0]
        # img = batch['img'] # 3 x H x W
        pose_gt = batch['pose']
        batch_size = pts_2d_gt.shape[0]

        open_cv_image_path = os.path.join(data_root, dataset, f"{data_id}.rgb.jpg")        
        open_cv_image = cv2.imread(open_cv_image_path)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)


        robot = PandaRobotArm(urdf_path="urdfs/Panda/panda.urdf", ee_name="panda_hand", device='cpu')
        _, pts_3d_robot = robot.get_T_all_links(joint_angles)
        pts_3d_robot = pts_3d_robot[:, [0,1,2,3,4,5,6,7]]
        pts_3d_robot[:, -2, 0] += 0.0825
        pts_3d_robot[:, -1, -1] -= 0.088

        R = pose_gt[:, :3, :3]
        t = pose_gt[:, :3, 3]

        pts_3d_obj_hom = kornia.geometry.conversions.convert_points_to_homogeneous(pts_3d_robot)
        pts_3d_cam = pose_gt @ pts_3d_obj_hom.transpose(-1, -2)
        pts_3d_cam = pts_3d_cam[:, :3, :] # remove homogeneous coordinate
        pts_2d = kornia.geometry.camera.perspective.project_points(pts_3d_cam.transpose(-1, -2), K) #pts_3d_cam.transpose(-1, -2) za col-major
        pts_2d = pts_2d.int()

        # pts_3d_cam = R @ pts_3d_robot.transpose(-1, -2) + t.unsqueeze(-1) # (bs, 3, N)
        # pts_3d_proj = (K @ pts_3d_cam).transpose(-1, -2) # (bs, N, 3)
        # pts_2d = (pts_3d_proj / pts_3d_proj[..., -1].unsqueeze(-1))[..., :2] # (bs, N, 3) / (bs, N, 1) -> (bs, N, 2)
        # pts_2d = pts_2d.int()

        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,0]]), tuple([coord.item() for coord in pts_2d[0,1]]), (255, 0, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,1]]), tuple([coord.item() for coord in pts_2d[0,3]]), (0, 255, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,3]]), tuple([coord.item() for coord in pts_2d[0,4]]), (0, 0, 255), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,4]]), tuple([coord.item() for coord in pts_2d[0,5]]), (255, 255, 0), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,5]]), tuple([coord.item() for coord in pts_2d[0,6]]), (0, 255, 255), thickness=5)
        open_cv_image = cv2.line(open_cv_image, tuple([coord.item() for coord in pts_2d[0,6]]), tuple([coord.item() for coord in pts_2d[0,7]]), (255, 0, 255), thickness=5)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"skeleton_{i}.jpg", open_cv_image)


if __name__ == "__main__":
    test_draw_skeleton()
