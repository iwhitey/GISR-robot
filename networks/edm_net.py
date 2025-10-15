import torch
from torch import nn
from torch.nn.functional import normalize

import numpy as np
from numpy.linalg import norm   
from networks.edm_ops import batch_joint_variables, solve_mds, symmetric_from_upper_triu_values, get_constant_data
import sys

try:
    import graphik
except:
    sys.path.append("../graphIK")
try:
    import liegroups
except:
    sys.path.append("../liegroups")
try:
    import urdfpy
except:
    sys.path.append("../urdfpy")

import graphik
from graphik.utils.roboturdf import load_panda


class FullyConnectedNetwork(nn.Module):
    def __init__(self, num_keypoints, hidden_size, complete_edm):
        super(FullyConnectedNetwork, self).__init__()

        self.num_points = 2*num_keypoints+2
        self.in_dim = (num_keypoints * (num_keypoints - 1)) // 2
        self.out_dim = self.in_dim if not complete_edm else (self.num_points * (self.num_points - 1)) // 2
        self.robot, _ = load_panda()
        self.M, _= get_constant_data(self.robot)

        self.model= nn.Sequential(
            nn.Linear(self.in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(hidden_size, self.out_dim),
            nn.ReLU(inplace=True)
        )

        self.model.apply(init_weights)
    
    def forward(self, x):
        model_output = self.model(x)
        D_pred = symmetric_from_upper_triu_values(model_output, self.num_points)
        P = solve_mds(D_pred)
        q_pred = batch_joint_variables(P, self.M, self.robot.n)[:, :6].float()
        return q_pred

def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        module.bias.data.fill_(0.01)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)