import os
import argparse
import torch

from eval.eval import evaluate
from eval.eval_ur5 import evaluate_ur5
from config import get_eval_cfg
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test')
    parser.add_argument('--robot', choices=['ur5', 'panda'], type=str)
    args = parser.parse_args()
    cfg = get_eval_cfg(args)
    
    if cfg.robot == "panda":
        evaluate(cfg)
    elif cfg.robot == "ur5":
        evaluate_ur5(cfg)