import sys, os
import json

import torch
import pytorch_kinematics as pk

from utils.pose_utils import UR5_Rotation

import numpy as np
import math


class PandaRobotArm:

    def __init__(self, urdf_path, ee_name="lbr_iiwa_link_7", device=torch.device("cpu")):

        self.joint_01_length = 0.333
        self.joint_23_length = 0.316
        self.joint_45_length = 0.384

        self.device = device
        self.urdf_path = urdf_path
        self.chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), ee_name)
        self.chain = self.chain.to(device=device)
        self.link_ids = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand']
        PandaRobotArm.joint_limits_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=torch.float32, device=device)
        PandaRobotArm.joint_limits_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=torch.float32, device=device)
    
    def get_joint_3d_locations_from_angles(self, joint_angles):
        _, t_robot = self.get_T_all_links(joint_angles)
        pts_3d_obj = t_robot[:, [1,2,3,4,5,6,7]]
        return pts_3d_obj

    def get_T_all_links(self, joint_angles):
        if joint_angles.shape[1] == 6:
            ee_angle = self.joint_limits_min[-1] + 1e-3 + (self.joint_limits_max[-1] - 1e-3 - (self.joint_limits_min[-1] + 1e-3)) * torch.rand((joint_angles.shape[0], 1), device=joint_angles.device)
            joint_angles = torch.cat([joint_angles, ee_angle], dim=1)
        T_all_links = self.chain.forward_kinematics(joint_angles, end_only=False)
        R_list = []; t_list = []
        for link in self.link_ids:
            T = T_all_links[link].get_matrix()
            R_list.append(T[:, :3, :3])
            t_list.append(T[:, :3, 3])
        
        R = torch.stack(R_list, dim=1)
        t = torch.stack(t_list, dim=1)
        return R, t
    

class UR5RobotArm:

    def __init__(self, urdf_path, ee_name="wrist_3_link", device=torch.device("cpu")):

        self.device = device
        self.urdf_path = urdf_path
        self.chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), ee_name)
        self.chain = self.chain.to(device=device)
        UR5RobotArm.joint_limits_min = torch.tensor([-math.pi/2, -math.pi, -math.pi/2, -math.pi, -math.pi/2, -math.pi], dtype=torch.float32, device=device)
        UR5RobotArm.joint_limits_max = torch.tensor([math.pi/4, 0., math.pi/2, 0., math.pi/2, math.pi], dtype=torch.float32, device=device)
        self.rot_xz = UR5_Rotation.rot_xz.to(device=device)

    def get_T_all_links(self, joint_angles):

        T_all_links = self.chain.forward_kinematics(joint_angles, end_only=False)
        R_list = []; t_list = []
        for link in T_all_links.keys():
            T = T_all_links[link].get_matrix()
            R_list.append(T[:, :3, :3])
            t_list.append(T[:, :3, 3])
        
        R = torch.stack(R_list, dim=1)
        t = torch.stack(t_list, dim=1)
        return R, t

    
    def get_joint_3d_locations_from_angles(self, joint_angles):
        _, t_robot = self.get_T_all_links(joint_angles)
        pts_3d_obj = t_robot[:, 1:, :]
        return pts_3d_obj
    
    def get_joint_2d_locations_from_angles(self, joint_angles, camera, image_size):
        pts_3d = self.get_joint_3d_locations_from_angles(joint_angles)
        pts_2d = camera.transform_points_screen(pts_3d, image_size = image_size)[:, :, :2]
        return pts_2d
    
    def get_joint_3d_locations_wrt_camera(self, joint_angles, R, t):
        pts_3d_robot = self.get_joint_3d_locations_from_angles(joint_angles)
        pts_3d_cam = pts_3d_robot @ R + t[:, None, :] # R is row-wise
        return pts_3d_cam
    

class BaxterRobotArm:

    def __init__(self, urdf_path, ee_name="left_wrist", device=torch.device("cpu")):

        self.device = device
        self.urdf_path = urdf_path
        self.chain = pk.build_serial_chain_from_urdf(open(urdf_path).read(), end_link_name=ee_name)
        self.chain = self.chain.to(device=device)
        # 'left_arm_mount' was the first link
        self.link_ids = ['left_upper_shoulder', 'left_lower_shoulder', 'left_upper_elbow', 'left_lower_elbow', 'left_upper_forearm', 'left_lower_forearm', 'left_wrist']
        BaxterRobotArm.joint_limits_min = torch.tensor([0.89, 1.05, 3.02, 2.62, 3.05, 2.09, 3.05], dtype=torch.float32, device=device)
        BaxterRobotArm.joint_limits_max = torch.tensor([-2.46, -2.15, -3.02, -0.05, -3.05, -1.57, -3.05], dtype=torch.float32, device=device)
        
    def get_T_all_links(self, joint_angles):
        if joint_angles.shape[1] == 6:
            ee_angle = self.joint_limits_min[-1] + 1e-3 + (self.joint_limits_max[-1] - 1e-3 - (self.joint_limits_min[-1] + 1e-3)) * torch.rand((joint_angles.shape[0], 1), device=joint_angles.device)
            joint_angles = torch.cat([joint_angles, ee_angle], dim=1)

        T_all_links = self.chain.forward_kinematics(joint_angles, end_only=False)
        R_list = []; t_list = []
        for link in self.link_ids:
            T = T_all_links[link].get_matrix()
            R_list.append(T[:, :3, :3])
            t_list.append(T[:, :3, 3])
        
        R = torch.stack(R_list, dim=1)
        t = torch.stack(t_list, dim=1)
        return R, t

    

if __name__ == "__main__":
    pass