import torch
import torch.nn as nn

from torchvision.models import convnext
from torchvision.models.convnext import ConvNeXt_Tiny_Weights

from networks.convnext_utils import LayerNorm2d, GradNorm


class ConvNext(nn.Module):
    def __init__(self, network_config, heads = ["q", "pose"]):
        super().__init__()

        self.network_config = network_config
        if network_config["type"] == "convnext_tiny":
            self.backbone = convnext.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.lastconv_output_channels = 768
            self.clasifier = None

        conv_0_0_weights = self.backbone.features[0][0].weight
        new_input_conv_layer = nn.Conv2d(network_config["in_channels"], 96, kernel_size=(4, 4), stride=(4, 4)) # 6 or 4 or 2 channels
        if network_config["in_channels"] == 2:
            new_input_conv_layer_weights = torch.cat(((conv_0_0_weights.mean(dim=1, keepdim=True)), (conv_0_0_weights.mean(dim=1, keepdim=True))), dim=1) # for 2 channels
        else:
            new_input_conv_layer_weights = conv_0_0_weights.repeat(1, 2, 1, 1) # for 6 channels
            if network_config["in_channels"] == 4:
                new_input_conv_layer_weights = torch.cat((conv_0_0_weights, (conv_0_0_weights.mean(dim=1, keepdim=True))), dim=1) # for 4 channels

        with torch.no_grad():
            new_input_conv_layer.weight[:, :new_input_conv_layer_weights.shape[1], :, :] = new_input_conv_layer_weights.clone()
        self.backbone.features[0][0] = new_input_conv_layer
        # print("new weights shape: ", new_input_conv_layer_weights.shape); print("new conv layer weights shape: ", new_input_conv_layer.weight.shape)

        self.q_head = nn.Linear(self.lastconv_output_channels, network_config["q_out_dim"])
        self.pose_head = nn.Linear(self.lastconv_output_channels, 7) # 6D rot parametrization + scale correction
        init_weights([self.q_head, self.pose_head])


    def forward(self, imgs):
        x = self.backbone.features(imgs)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        q_update = self.q_head(GradNorm.apply(x))
        pose_update = self.pose_head(GradNorm.apply(x))
        return q_update, pose_update
    

def init_weights(modules):
    for module in modules:
        print(f"Initializing weights for module: {module}")
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            module.bias.data.fill_(0.01)
        elif isinstance(module, (LayerNorm2d, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)