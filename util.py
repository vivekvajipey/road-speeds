import torch
import torchvision
import numpy as np
import pandas as pd
import timm
from torchgeo.models import ResNet18_Weights, ResNet50_Weights

import random
from typing import Optional


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_resnet(hook: str, pretrain: Optional[str]):
    if pretrain in ['sent2-moco-all', 'setn2-moco-rgb']:
        if pretrain == 'sent2-moco-all':
            model = timm.create_model(hook, in_chans=13, num_classes=1)
            if hook == 'resnet18':
                weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
            elif hook == 'resnet50':
                weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
            else:
                raise ValueError('Moco pretraining only valid for resnet 18 and 50')
        elif pretrain == 'sent2-moco-rgb':
            model = timm.create_model(hook, in_chans=3, num_classes=1)
            if hook == 'resnet18':
                weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
            elif hook == 'resnet50':
                weights = ResNet50_Weights.SENTINEL2_RGB_MOCO
            else:
                raise ValueError('Moco pretraining only valid for resnet 18 and 50')
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model.relu = model.act1 # rename for compatibility
        model.avgpool = model.global_pool
        return model
    weights = None
    if pretrain == 'imagenet':
        weights = 'DEFAULT'
    if hook == 'resnet18':
        return torchvision.models.resnet18(weights=weights)
    elif hook == 'resnet34':
        return torchvision.models.resnet34(weights=weights)
    elif hook == 'resnet50':
        return torchvision.models.resnet50(weights=weights)
    else:
        raise ValueError(f'Model name {hook} not found.')

def get_channel_dim(channels: list[str]):
    dim = 0
    for channel in channels:
        if channel == 'rgb_path':
            dim += 3
        elif channel == 'nir_path':
            dim += 1
    return dim

def get_tabular_dim(cols: list[str]):
    dim = 0
    for col in cols:
        if col == 'year':
            dim += 3
        if col == 'month':
            dim += 11
        elif col == 'hour_ratios':
            dim += 24
        else:
            dim += 1
    return dim
