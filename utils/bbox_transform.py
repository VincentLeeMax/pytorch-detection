import torch

def to_cornor_form(anchors, newone=True, image_shape=None):
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

    # if image_shape is not None:
    #     twopoint_anchors[:, :, 0::2] = torch.clamp(twopoint_anchors[:, :, 0::2], 0, image_shape[1])
    #     twopoint_anchors[:, :, 1::2] = torch.clamp(twopoint_anchors[:, :, 1::2], 0, image_shape[0])

    return twopoint_anchors


def to_central_form(anchors, newone=True):
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


def regress_box(anchors, regressions, newone=True):
    if newone:
        regressed_anchors = torch.ones(anchors.shape) * 1.0
    else:
        regressed_anchors = anchors

    if regressions.shape[0] > 1:
        regressed_anchors = regressed_anchors.expand_as(regressions).clone()

    if torch.cuda.is_available() and newone:
        regressed_anchors = regressed_anchors.cuda()

    regressed_anchors[:, :, 0] = anchors[:, :, 2] * regressions[:, :, 0] + anchors[:, :, 0]
    regressed_anchors[:, :, 1] = anchors[:, :, 3] * regressions[:, :, 1] + anchors[:, :, 1]
    regressed_anchors[:, :, 2] = anchors[:, :, 2] * torch.exp(regressions[:, :, 2])
    regressed_anchors[:, :, 3] = anchors[:, :, 3] * torch.exp(regressions[:, :, 3])

    return regressed_anchors