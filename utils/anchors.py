import torch
import torch.nn as nn

import numpy as np

class FPNAnchors(nn.Module):
    def __init__(self, fpn_levels=None, level_strides=None, level_scales=None, anchor_sizes=None, anchor_ratios=None):
        super(FPNAnchors, self).__init__()
        if fpn_levels is None:
            self.fpn_levels = [3, 4, 5, 6, 7]
        if level_strides is None:
            self.level_strides = [2 ** x for x in self.fpn_levels]
        if level_scales is None:
            self.level_sizes = [2 ** (x + 2) for x in self.fpn_levels]
        if anchor_sizes is None:
            self.anchor_scales = [2 ** 0, 2 ** (1/3), 2 ** (2/3)]
        if anchor_ratios is None:
            self.anchor_ratios = [0.5, 1, 2]
        
    def forward(self, imgs):
        image_shape = np.array(imgs.shape[-2:])
        feature_shape = [np.ceil(image_shape * 1. / stride) for stride in self.level_strides]
        fpn_anchors = np.zeros((0, 4), dtype=np.float)
        for idx, fpn_level in enumerate(self.fpn_levels):
            ### generate standard anchor
            anchor_num = len(self.anchor_scales) * len(self.anchor_ratios)
            standard_anchors = np.zeros((anchor_num, 4), dtype=np.float)
            standard_anchors[:, -2:] = np.tile(self.anchor_scales, (2, len(self.anchor_ratios))).T
            
            # height&width(ratio * x, x) : ratio * x^2 = level_size * scale
            standard_anchors[:, 3] = np.sqrt(self.level_sizes[idx] * np.repeat(self.anchor_scales, len(self.anchor_ratios)) * standard_anchors[:, 3])
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
            tiled_anchors = np.tile(tiled_anchors, (anchor_num, 1))
            # append height width
            tiled_anchors = np.hstack((tiled_anchors, np.tile(standard_anchors[:, -2:], (tiled_anchors.shape[0] // anchor_num, 1))))
            # modify centre
            tiled_anchors[:, 0] = (tiled_anchors[:, 0] + 0.5) * self.level_strides[idx]
            tiled_anchors[:, 1] = (tiled_anchors[:, 1] + 0.5) * self.level_strides[idx]
            
            fpn_anchors = np.append(fpn_anchors, tiled_anchors, axis=0)
        
        fpn_anchors = np.expand_dims(fpn_anchors, axis=0)
        
        # clamp the boundary box
        fpn_anchors = torch.from_numpy(fpn_anchors.astype(np.float32))
        fpn_anchors[:, :, 0] = fpn_anchors[:, :, 0] - fpn_anchors[:, :,2] / 2
        fpn_anchors[:, :, 2] = fpn_anchors[:, :, 0] + fpn_anchors[:, :,2]
        fpn_anchors[:, :, 1] = fpn_anchors[:, :, 1] - fpn_anchors[:, :,3] / 2
        fpn_anchors[:, :, 3] = fpn_anchors[:, :, 1] + fpn_anchors[:, :,3]
        
        torch.clamp(fpn_anchors[:, :, 0::2], 0, image_shape[1])
        torch.clamp(fpn_anchors[:, :, 1::2], 0, image_shape[0])
        
        fpn_anchors[:, :, 2] = fpn_anchors[:, :, 2] - fpn_anchors[:, :, 0]
        fpn_anchors[:, :, 0] = fpn_anchors[:, :, 0] + fpn_anchors[:, :, 2] / 2
        fpn_anchors[:, :, 3] = fpn_anchors[:, :, 3] - fpn_anchors[:, :, 1]
        fpn_anchors[:, :, 1] = fpn_anchors[:, :, 1] + fpn_anchors[:, :, 3] / 2
        
        return fpn_anchors
    

if __name__ == '__main__':
    x = np.random.rand(2, 3, 700, 1000)
    model = FPNAnchors()
    model(x)