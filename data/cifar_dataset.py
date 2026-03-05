import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np
import cv2


class CIFARColorization(Dataset):
    def __init__(self, root, train=True):
        self.dataset = CIFAR10(
            root=root,
            train=train,
            download=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx] 

        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

        L = lab[:, :, 0] / 255.0 # черно=белые изображения
        ab = lab[:, :, 1:] / 128.0 - 1.0

        L = torch.tensor(L).unsqueeze(0)        
        ab = torch.tensor(ab).permute(2, 0, 1)

        return L, ab
