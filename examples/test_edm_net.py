import sys
import torch
from networks.edm_net import FullyConnectedNetwork
sys.path.append("../graphIK")
sys.path.append("../liegroups")
sys.path.append("../urdfpy")


if __name__ == "__main__":
    model = FullyConnectedNetwork(7, 512, complete_edm=True)
    edm_2d_upper = torch.randn((128, 21)).float()
    q = model(edm_2d_upper)
    print(q)