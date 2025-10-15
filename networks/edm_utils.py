import torch
from networks.edm_net import FullyConnectedNetwork

def set_edm_model_eval(model):
    """Set EDM model to eval mode based on single/multi-GPU setup"""
    if hasattr(model, 'module'):
        # Multi-GPU (DDP wrapped)
        if hasattr(model.module, 'edm_model'):
            model.module.edm_model.eval()
    else:
        # Single-GPU
        if hasattr(model, 'edm_model'):
            model.edm_model.eval()

def add_edm_model_to_state_dict(state_dict, edm_model_path):
    if "edm_model.model.0.weight" not in state_dict.keys():
        edm_state_dict = torch.load(edm_model_path)['model_state_dict']
        for k, v in edm_state_dict.items():
            name = "edm_model."+k
            state_dict[name] = v
    return state_dict

def get_edm_model(model_path, num_keypoints = 7, hidden_size = 512, complete_edm = True):
    model = FullyConnectedNetwork(num_keypoints, hidden_size, complete_edm)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    return model