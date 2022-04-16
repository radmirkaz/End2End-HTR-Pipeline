from abc import ABC

import torch
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image, ImageDraw
import cv2
import PIL
import random
from albumentations.core.transforms_interface import ImageOnlyTransform
import matplotlib.pyplot as plt
# from fmix import sample_mask # Not Implemented
from matplotlib import cm

def to_tensorv2(p: int = 1.0):
    return ToTensorV2(p=p)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.clip(np.random.beta(alpha, alpha), 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    targets = {
        'target': target,
        'shuffled_target': shuffled_target,
        'lam': lam
    }

    return new_data, targets


# def fmix(data, target, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
#     lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)

#     indices = torch.randperm(data.size(0))
#     shuffled_data = data[indices]
#     shuffled_target = target[indices]

#     x1 = torch.from_numpy(mask).to(data.device) * data
#     x2 = torch.from_numpy(1 - mask).to(data.device) * shuffled_data
#     targets = {
#         'target': target,
#         'shuffled_target': shuffled_target,
#         'lam': lam
#     }

#     return x1 + x2, targets


def mixup(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha) # no clipping to take a whole image
    new_data = data.clone()
    new_data = data * lam + shuffled_data * (1.0 - lam)
    targets = {
        'target': target,
        'shuffled_target': shuffled_target,
        'lam': lam
    }

    return new_data, targets


class ExtraLinesAugmentation(ImageOnlyTransform):
    '''
    Add random black lines to an image
    Args:
        number_of_lines (int): number of black lines to add
        width_of_lines (int): width of lines
    '''

    def __init__(self, number_of_lines: int = 1, width_of_lines: int = 10, p=0.5):
        super(ExtraLinesAugmentation, self).__init__(True, p)
        self.number_of_lines = number_of_lines
        self.width_of_lines = width_of_lines

    def apply(self, img, **params):
        '''
        Args:
          img (PIL Image): image to draw lines on
        Returns:
          PIL Image: image with drawn lines
        '''

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for _ in range(self.number_of_lines):
            x1 = random.randint(0, np.array(img).shape[1])
            y1 = random.randint(0, np.array(img).shape[0])
            x2 = random.randint(0, np.array(img).shape[1])
            y2 = random.randint(0, np.array(img).shape[0])
            draw.line((x1, y1, x2, y2), fill=0, width=self.width_of_lines)

        return np.array(img)

    def get_transform_init_args_names(self):
        return (
            "number_of_lines",
            "width_of_lines",
        )

    def get_params(self):
        return {
            "number_of_lines": self.number_of_lines,
            "width_of_lines": self.width_of_lines,
        }

