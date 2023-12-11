import torch
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image
import pandas as pd
import numpy as np

import os
from typing import Callable, Optional


# Create custom dataset and dataloaders
class RoadMeanSTDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        image_cols: list[str] = ['rgb_path'],
        image_transform: Optional[Callable] = None,
        tabular_cols: Optional[list[str]] = None,
        tabular_transform: Optional[dict[str, Callable]] = None,
        split: str = 'train',
    ):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.image_cols = image_cols
        self.image_transform = image_transform
        self.tabular_info = []
        if tabular_cols is not None:
            self.tabular_df = self.df[tabular_cols]
            for col in tabular_cols:
                if tabular_transform is not None and col in tabular_transform:
                    xform = tabular_transform[col]
                    self.tabular_info.append(xform(self.tabular_df[col]))
                else:
                    self.tabular_info.append(self.tabular_df[col].to_numpy()[:, None])
        self.tabular_info = np.concatenate(self.tabular_info, axis=1)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_stack = []
        for image_col in self.image_cols:
            img_path = self.df.loc[idx, image_col]
            img = pil_to_tensor(Image.open(img_path)) # channels first
            img_stack.append(img)
        img_full = torch.cat(img_stack, dim=0)
        img_full = self.image_transform(img_full) # (n_channels, 224, 224)

        tab_data = torch.tensor(self.tabular_info[idx], dtype=torch.float32) # (n_feats,)

        # mean_std_target = self.df.loc[idx, ['speed_mean', 'speed_std']].to_numpy()
        mean_std_target = self.df.loc[idx, ['speed_mean_norm', 'speed_std_norm']].to_numpy()
        mean_std_target = mean_std_target.astype(np.float32)
        mean_std_target = torch.tensor(mean_std_target, dtype=torch.float32) # (2,)

        return img_full, tab_data, mean_std_target