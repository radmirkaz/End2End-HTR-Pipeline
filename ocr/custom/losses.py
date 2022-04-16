import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0

    input_flat = input.view(-1)
    target_flat = target.view(-1)

    intersection = (input_flat * target_flat).sum()

    return ((2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()

        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()

        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(dice_loss(input, target))

        return loss.mean()