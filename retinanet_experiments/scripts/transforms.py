import random
from typing import Dict, Tuple

import torch
from torchvision.transforms import functional as F


class Compose:
    """Compose multiple transforms that operate on image & target pairs."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """Convert a PIL image to a float tensor in [0, 1]."""

    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Randomly flip the image and bounding boxes horizontally."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target: Dict[str, torch.Tensor]):
        if random.random() < self.prob:
            image = F.hflip(image)
            width = target["size"][1].item()

            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                xmin = boxes[:, 0]
                xmax = boxes[:, 2]
                boxes[:, 0] = width - xmax
                boxes[:, 2] = width - xmin
                target["boxes"] = boxes
        return image, target


def build_transforms(train: bool, enable_hflip: bool = True):
    tx = []
    if train and enable_hflip:
        tx.append(RandomHorizontalFlip(prob=0.5))
    tx.append(ToTensor())
    return Compose(tx)

