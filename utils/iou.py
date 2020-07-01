import torch


def batch_cal_iou(bboxs, targets):
    area_bboxs = (bboxs[:, 3] - bboxs[:, 1]) * (bboxs[:, 2] - bboxs[:, 0])
    area_targets = (targets[:, 3] - targets[:, 1]) * (targets[:, 2] - targets[:, 0])

    intersect_w = torch.min(torch.unsqueeze(bboxs[:, 2], dim=1), targets[:, 2]) - torch.max(
        torch.unsqueeze(bboxs[:, 0], dim=1), targets[:, 0])
    intersect_h = torch.min(torch.unsqueeze(bboxs[:, 3], dim=1), targets[:, 3]) - torch.max(
        torch.unsqueeze(bboxs[:, 1], dim=1), targets[:, 1])
    intersect_area = torch.clamp(intersect_w, 0) * torch.clamp(intersect_h, 0)

    ious = intersect_area / (torch.unsqueeze(area_bboxs, dim=1) + area_targets - intersect_area)

    return ious


if __name__ == '__main__':
    import numpy as np
    from utils.bbox_transform import to_cornor_form

    bboxs = torch.from_numpy(np.array([[157.2000, 168.0000, 193.8000, 270.4000,   9.0000],
        [ 98.4000, 210.4000, 151.2000, 296.8000,   9.0000],
        [144.0000, 154.4000, 176.4000, 238.4000,   9.0000]])).float()
    print(bboxs)
    targets = torch.from_numpy(np.array([[157.2000, 168.0000, 193.8000, 270.4000,   9.0000],
        [ 98.4000, 210.4000, 151.2000, 296.8000,   9.0000],
        [144.0000, 154.4000, 176.4000, 238.4000,   9.0000]])).float()
    print(targets)
    ious = batch_cal_iou(bboxs, targets)

    print(ious)
    print(torch.max(ious, dim=1))
