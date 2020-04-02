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

    bboxs = torch.from_numpy(np.array([[0, 278, 906, 424]])).float()
    targets = torch.from_numpy(np.array([[ 0.0000,   91.9299,  776.1401,  548.0701]])).float()
    ious = batch_cal_iou(bboxs, targets)

    print(ious)
