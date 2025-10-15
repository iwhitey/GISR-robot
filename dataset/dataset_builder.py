import os, sys
import torch
from dataset.dataset import UR5SynthDataset, EDMDatasetReal, EDMDatasetSynthetic

class DatasetFactory:
    def __init__(self, robot, is_train, **kwargs):
        self.robot = robot
        self.is_train = is_train
        self.kwargs = kwargs

    def create(self):
        if self.robot == "panda":
            dataset = self.build_panda()
        elif self.robot == "ur5":
            dataset = self.build_ur5()
        else:
            raise ValueError("Unknown robot")
        return dataset
    
    def build_panda(self):
        if isinstance(self.kwargs["dataset"], list):
            datasets = []
            for dataset_i in self.kwargs["dataset"]:
                datasets.append(EDMDatasetReal(self.kwargs["data_root"], dataset_i, self.kwargs["complete_edm"], 
                                               self.robot, self.kwargs["cfg"]))
            dataset = torch.utils.data.ConcatDataset(datasets)

        else:
            if "synth" not in self.kwargs["dataset"]:
                dataset = EDMDatasetReal(self.kwargs["data_root"], self.kwargs["dataset"], self.kwargs["complete_edm"], 
                                               self.robot, self.kwargs["cfg"])
            else:
                dataset = EDMDatasetSynthetic(self.kwargs["data_root"], self.kwargs["dataset"], self.kwargs["complete_edm"], 
                                               self.robot, self.kwargs["cfg"])
        return dataset
    
    def build_ur5(self):
        dataset = UR5SynthDataset(self.kwargs["data_root"], self.kwargs["dataset"], self.kwargs["cfg"])
        return dataset