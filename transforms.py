import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as xforms

import json

class ConditionalNormalize(xforms.Transform):
    def __init__(self, channel_means, channel_stds):
        super().__init__()
        self.channel_means = channel_means
        self.channel_stds = channel_stds

    def _transform(self, inpt, params=None):
        if inpt.shape[0] == 3:
            return xforms.functional.normalize(inpt, mean=self.channel_means[:3], std=self.channel_stds[:3])
        else:
            return xforms.functional.normalize(inpt, mean=self.channel_means, std=self.channel_stds)

class RGBOnlyTransforms(xforms.Transform):
    def __init__(self, transforms: list):
        super().__init__()
        self.transforms = transforms

    def _transform(self, input, params=None):
        for transform in self.transforms:
            input[:3] = transform(input[:3])
        return input

TABULAR_TRANSFORMS = {
    'year': lambda x: pd.get_dummies(x).to_numpy().astype(np.float32),
    'month': lambda x: pd.get_dummies(x).reindex(columns=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], fill_value=0).to_numpy().astype(np.float32),
    'hour_ratios': lambda x: np.array(x.apply(json.loads).to_list()).astype(np.float32),
}

TRAIN_TRANSFORMS = xforms.Compose([
    RGBOnlyTransforms([
        xforms.ColorJitter(brightness=(0.5, 2), contrast=0.2, saturation=0.2, hue=0.2),
    ]),
    xforms.RandomHorizontalFlip(),
    xforms.RandomVerticalFlip(),
    xforms.RandomApply([xforms.RandomRotation((90, 90))], p=0.5),
    xforms.ToDtype(torch.float32, scale=True),
    ConditionalNormalize(
        [0.0171, 0.0154, 0.0082, 0.0674], # R,G,B,NIR
        [0.0101, 0.0129, 0.0143, 0.1278]
    ),
])

VAL_TRANSFORMS = xforms.Compose([
    xforms.ToDtype(torch.float32, scale=True),
    ConditionalNormalize(
        [0.0171, 0.0154, 0.0082, 0.0674],
        [0.0101, 0.0129, 0.0143, 0.1278]
    ),
])