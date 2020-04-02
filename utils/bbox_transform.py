import torch


def to_twoPoint_format(anchors, newone=False, image_shape=None):
    if newone:
        twopoint_anchors = torch.ones(anchors.shape) * 1.0
    else:
        twopoint_anchors = anchors
    if torch.cuda.is_available() and newone:
        twopoint_anchors = twopoint_anchors.cuda()

    twopoint_anchors[:, :, 0] = anchors[:, :, 0] - anchors[:, :, 2] / 2
    twopoint_anchors[:, :, 2] = twopoint_anchors[:, :, 0] + anchors[:, :, 2]
    twopoint_anchors[:, :, 1] = anchors[:, :, 1] - anchors[:, :, 3] / 2
    twopoint_anchors[:, :, 3] = twopoint_anchors[:, :, 1] + anchors[:, :, 3]

    if image_shape is not None:
        twopoint_anchors[:, :, 0::2] = torch.clamp(twopoint_anchors[:, :, 0::2], 0, image_shape[1])
        twopoint_anchors[:, :, 1::2] = torch.clamp(twopoint_anchors[:, :, 1::2], 0, image_shape[0])
    else:
        twopoint_anchors[:, :, 0::2] = torch.clamp(twopoint_anchors[:, :, 0::2], 0)
        twopoint_anchors[:, :, 1::2] = torch.clamp(twopoint_anchors[:, :, 1::2], 0)

    return twopoint_anchors


def to_central_format(anchors, newone=False):
    if newone:
        central_anchors = torch.ones(anchors.shape) * 1.0
    else:
        central_anchors = anchors
    if torch.cuda.is_available() and newone:
        central_anchors = central_anchors.cuda()

    central_anchors[:, :, 2] = anchors[:, :, 2] - anchors[:, :, 0]
    central_anchors[:, :, 0] = anchors[:, :, 0] + central_anchors[:, :, 2] / 2
    central_anchors[:, :, 3] = anchors[:, :, 3] - anchors[:, :, 1]
    central_anchors[:, :, 1] = anchors[:, :, 1] + central_anchors[:, :, 3] / 2

    return central_anchors
