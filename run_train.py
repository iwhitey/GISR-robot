import os
import argparse
import torch

from train.train import train_iter
from train.train_segmentation import train_seg

import config


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('--robot', choices=['ur5', 'panda'], type=str)
    args = parser.parse_args()
    train_cfg = config.get_train_cfg(args)    
    train_iter(train_cfg)