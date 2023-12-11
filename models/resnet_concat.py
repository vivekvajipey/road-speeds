import torch
import torch.nn as nn
import torchvision

from typing import Union, Any, Optional

class ResNetConcat(nn.Module):

    def __init__(
        self,
        resnet: nn.Module,
        channel_dim: int,
        tabular_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        hidden_dim: Optional[int] = None, # if none, no additional dim
        pretrain_type: Optional[str] = None,
    ):
        super().__init__()

        self.resnet = resnet
        if pretrain_type == 'imagenet':
            conv1_weight_original = resnet.conv1.weight.clone()
            self.resnet.conv1 = nn.Conv2d(channel_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv1_weight_new = self.resnet.conv1.weight.data.clone()
            conv1_weight_new[:, :3] = conv1_weight_original
            self.resnet.conv1.weight.data = conv1_weight_new
        elif pretrain_type == 'sent2-moco-all': # maybe more efficient way than below but oh well
            conv1_rgb_weight = resnet.conv1.weight[:, [3,2,1]].clone()
            conv1_nir_weight = resnet.conv1.weight[:, 7].clone()
            self.resnet.conv1 = nn.Conv2d(channel_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv1_weight_new = self.resnet.conv1.weight.data.clone()
            conv1_weight_new[:, :3] = conv1_rgb_weight
            conv1_weight_new[:, 3] = conv1_nir_weight
            self.resnet.conv1.weight.data = conv1_weight_new
        elif pretrain_type == 'sent2-moco-rgb':
            conv1_rgb_weight = resnet.conv1.weight[:, [2,1,0]].clone()
            self.resnet.conv1 = nn.Conv2d(channel_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
            conv1_weight_new = self.resnet.conv1.weight.data.clone()
            conv1_weight_new[:, :3] = conv1_rgb_weight
            self.resnet.conv1.weight.data = conv1_weight_new

        self.resnet_dropout = nn.Dropout(dropout)

        fc_layers = []
        resnet_emb_dim = resnet.fc.in_features
        if hidden_dim is not None:
            fc_layers.append(nn.Linear(resnet_emb_dim + tabular_dim, hidden_dim))
            fc_layers.append(nn.GELU())
            # fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(hidden_dim, out_dim))
        else:
            fc_layers.append(nn.Linear(resnet_emb_dim + tabular_dim, out_dim))
        self.fc_layers = nn.ModuleList(fc_layers)

    def forward(self, img, tabular):
        x = self.resnet.conv1(img)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = torch.cat([x, tabular], dim=1)
        x = self.resnet_dropout(x)
        for layer in self.fc_layers:
            x = layer(x)
        return x
