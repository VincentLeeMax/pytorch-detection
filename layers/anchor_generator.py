import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import torch
import torch.nn as nn

from utils.bbox_transform import to_central_form, to_cornor_form

class FPNAnchors(nn.Module):
    def __init__(self, fpn_levels=None, level_strides=None, level_sizes=None, anchor_scales=None, anchor_ratios=None):
        super(FPNAnchors, self).__init__()
        self.fpn_levels = fpn_levels
        self.level_strides = level_strides
        self.level_sizes = level_sizes
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        if fpn_levels is None:
            self.fpn_levels = [3, 4, 5, 6, 7]
        if level_strides is None:
            self.level_strides = [2 ** x for x in self.fpn_levels]
        if level_sizes is None:
            self.level_sizes = [2 ** (x + 2) for x in self.fpn_levels]
        if anchor_scales is None:
            self.anchor_scales = [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)]
        if anchor_ratios is None:
            self.anchor_ratios = [0.5, 1, 2]

    def forward(self, imgs):
        image_shape = np.array(imgs.shape[-2:])
        feature_shape = [np.ceil(image_shape * 1. / stride) for stride in self.level_strides]
        fpn_anchors = np.zeros((0, 4), dtype=np.float)
        for idx, fpn_level in enumerate(self.fpn_levels):
            ### generate standard anchor's height and width
            anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)
            standard_anchors = np.zeros((anchor_num, 4), dtype=np.float)
            standard_anchors[:, -2:] = np.tile(self.anchor_ratios, (2, len(self.anchor_scales))).T

            # height&width(ratio * x, x) : ratio * x^2 = level_size^2 * scale^2
            standard_anchors[:, 3] = np.sqrt(
                self.level_sizes[idx] ** 2 * np.repeat(self.anchor_scales, len(self.anchor_ratios)) ** 2 / standard_anchors[:, 3])
            standard_anchors[:, 2] *= standard_anchors[:, 3]

            standard_anchors[:, 1] += standard_anchors[:, 3] / 2
            standard_anchors[:, 0] += standard_anchors[:, 2] / 2

            ### tile on the feature map
            x_coord = np.arange(0, feature_shape[idx][1])
            y_coord = np.arange(0, feature_shape[idx][0])

            x_coord, y_coord = np.meshgrid(x_coord, y_coord)
            # stack coord
            tiled_anchors = np.vstack((x_coord.ravel(), y_coord.ravel())).transpose()
            # copy
            tiled_anchors = np.repeat(tiled_anchors, anchor_num, axis=0)
            # append height width
            tiled_anchors = np.hstack(
                (tiled_anchors, np.tile(standard_anchors[:, -2:], (tiled_anchors.shape[0] // anchor_num, 1))))
            # modify centre
            tiled_anchors[:, 0] = (tiled_anchors[:, 0] + 0.5) * self.level_strides[idx]
            tiled_anchors[:, 1] = (tiled_anchors[:, 1] + 0.5) * self.level_strides[idx]

            fpn_anchors = np.append(fpn_anchors, tiled_anchors, axis=0)

        fpn_anchors = np.expand_dims(fpn_anchors, axis=0)

        # clamp the boundary box
        fpn_anchors = torch.from_numpy(fpn_anchors.astype(np.float32))
        if torch.cuda.is_available():
            fpn_anchors = fpn_anchors.cuda()
        fpn_anchors = to_cornor_form(fpn_anchors, image_shape=image_shape)
        fpn_anchors = to_central_form(fpn_anchors)

        return fpn_anchors

class SSDAnchors(nn.Module):
    def __init__(self, level_strides=None, anchor_sizes=None, anchor_ratios=None, feature_sizes=None):
        super(SSDAnchors, self).__init__()
        self.level_strides = level_strides
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.feature_sizes = feature_sizes

        if self.level_strides is None:
            assert self.level_strides is None
        if self.anchor_sizes is None:
            assert self.anchor_sizes is None
        if self.anchor_ratios is None:
            assert self.anchor_ratios is None
        if self.feature_sizes is None:
            assert self.feature_sizes is None

    def forward(self, imgs):
        image_shape = np.array(imgs.shape[-2:])

        fpn_anchors = np.zeros((0, 4), dtype=np.float)
        for idx, feature_size in enumerate(self.feature_sizes):
            ### generate standard anchor's height and width
            anchor_num = len(self.anchor_sizes[idx])
            standard_anchors = np.zeros((anchor_num, 4), dtype=np.float)
            standard_anchors[:, -2:] = np.tile(self.anchor_ratios[idx], (2, 1)).T

            # height&width(ratio * x, x) : ratio * x^2 = anchor_size^2
            standard_anchors[:, 3] = np.sqrt(np.array(self.anchor_sizes[idx]) ** 2 / standard_anchors[:, 2])
            standard_anchors[:, 2] *= standard_anchors[:, 3]

            standard_anchors[:, 1] += standard_anchors[:, 3] / 2
            standard_anchors[:, 0] += standard_anchors[:, 2] / 2

            ### tile on the feature map
            x_coord = np.arange(0, feature_size[1])
            y_coord = np.arange(0, feature_size[0])

            x_coord, y_coord = np.meshgrid(x_coord, y_coord)
            # stack coord
            tiled_anchors = np.vstack((x_coord.ravel(), y_coord.ravel())).transpose()
            # copy
            tiled_anchors = np.repeat(tiled_anchors, anchor_num, axis=0)
            # append height width
            tiled_anchors = np.hstack(
                (tiled_anchors, np.tile(standard_anchors[:, -2:], (tiled_anchors.shape[0] // anchor_num, 1))))
            # modify centre
            tiled_anchors[:, 0] = (tiled_anchors[:, 0] + 0.5) * self.level_strides[idx]
            tiled_anchors[:, 1] = (tiled_anchors[:, 1] + 0.5) * self.level_strides[idx]

            fpn_anchors = np.append(fpn_anchors, tiled_anchors, axis=0)

        fpn_anchors = np.expand_dims(fpn_anchors, axis=0)

        # clamp the boundary box
        fpn_anchors = torch.from_numpy(fpn_anchors.astype(np.float32))
        if torch.cuda.is_available():
            fpn_anchors = fpn_anchors.cuda()
        fpn_anchors = to_cornor_form(fpn_anchors, image_shape=image_shape)
        fpn_anchors = to_central_form(fpn_anchors)

        return fpn_anchors

if __name__ == '__main__':
    x = np.random.rand(2, 3, 300, 300)
    model = FPNAnchors()
    model(x)

    cfg = {'backbone': 'vgg16_300',
           'feature_maps': [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
           'strides': [8, 16, 32, 64, 100, 300],
           'anchor_sizes': [[30, 60, 30, 30],
                            [60, 111, 60, 60, 60, 60],
                            [111, 162, 111, 111, 111, 111],
                            [162, 213, 162, 162, 162, 162],
                            [213, 264, 213, 213],
                            [264, 315, 264, 264]],
           'anchor_ratios': [[1, 1, 0.5, 2],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2],
                             [1, 1, 0.5, 2]],
           'box_per_locations': [4, 6, 6, 6, 4, 4]}

    model = SSDAnchors(level_strides=cfg['strides'],
                       anchor_sizes=cfg['anchor_sizes'],
                       anchor_ratios=cfg['anchor_ratios'],
                       feature_sizes=cfg['feature_maps'])
    model(x)