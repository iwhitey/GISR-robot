import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models

class RobotSegNet(nn.Module):
    def __init__(self):
        super(RobotSegNet, self).__init__()

        deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        deeplabv3_resnet50.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
        self.backbone = torch.nn.Sequential(list(deeplabv3_resnet50.children())[0])
        self.classifier = torch.nn.Sequential((list(deeplabv3_resnet50.children())[1]))


    def forward(self, img):
        resnet_out = self.backbone(img)['out']  # (B, 2048, H//8, W//8)
        x = self.classifier(resnet_out)
        mask = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        return mask
