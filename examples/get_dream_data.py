import os
import json
import torch
import numpy as np
import math
import pytorch_kinematics as pk

from robot import PandaRobotArm


class Sample:
    def __init__(self, keypoints_2d, joint_angles, T_all_links):
        self.keypoints_2d = keypoints_2d
        self.joint_angles = joint_angles
        self.T_all_links = T_all_links


def get_model_and_data(urdf_path, ee_name, samples_path, device):

    robot = PandaRobotArm(urdf_path, ee_name, device)
    robot.to(device)

    all_samples = os.listdir(samples_path)
    all_samples = [os.path.join(samples_path, sample) for sample in all_samples if ".json" in sample and "camera" not in sample]
    samples = []

    for label_path in all_samples:
        
        with open(label_path, "r") as json_file:
                data = json.load(json_file)

        joint_angle = np.array([data['sim_state']['joints'][0]['position'],
                                data['sim_state']['joints'][1]['position'],
                                data['sim_state']['joints'][2]['position'],
                                data['sim_state']['joints'][3]['position'],
                                data['sim_state']['joints'][4]['position'],
                                data['sim_state']['joints'][5]['position'],
                                data['sim_state']['joints'][6]['position']])
        
        data['objects'][0]['keypoints'][0]['projected_location']
        keypoints_2d = np.array([data['objects'][0]['keypoints'][0]['projected_location'],
                                data['objects'][0]['keypoints'][1]['projected_location'],
                                data['objects'][0]['keypoints'][2]['projected_location'],
                                data['objects'][0]['keypoints'][3]['projected_location'],
                                data['objects'][0]['keypoints'][4]['projected_location'],
                                data['objects'][0]['keypoints'][5]['projected_location'],
                                data['objects'][0]['keypoints'][6]['projected_location']])
        
        keypoints_2d = torch.from_numpy(keypoints_2d).float().to(device)
        joint_angles = torch.from_numpy(joint_angle).float().to(device)
        T_all_links = robot.get_T_all_links(joint_angles)
        samples.append(Sample(keypoints_2d, joint_angles, T_all_links))

    return robot, samples