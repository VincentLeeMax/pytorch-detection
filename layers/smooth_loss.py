import torch
import torch.nn as nn

from utils.iou import batch_cal_iou


class SmoothL1(nn.Module):
    def __init__(self, sigma=1./9):
        super(SmoothL1, self).__init__()
        self.sigma = sigma

    def forward(self, input, target, reduction='none'):
        # smooth L1 loss
        regression_diff = torch.abs(input - target)
        regression_diff = torch.where(regression_diff >= self.sigma, regression_diff - 0.5 * self.sigma,
                                      torch.pow(regression_diff, 2) * 0.5 / self.sigma)

        if reduction != 'none':
            return regression_diff.mean() if reduction == 'mean' else regression_diff.sum()

        return regression_diff