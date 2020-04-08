import torch
import torch.nn as nn

from utils.IoU import batch_cal_iou
from utils.bbox_transform import to_twoPoint_format

class DetectionFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, detection_loss=None):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.detection_loss = detection_loss

    def forward(self, classifications, regressions, anchors, annotations):
        classification_losses = []
        regression_losses = []

        two_poinsts_anchors = to_twoPoint_format(anchors, newone=True)
        for batch_ in range(classifications.shape[0]):
            classification = classifications[batch_, :, :]
            regression = regressions[batch_, :, :]
            anchor = anchors[0, :, :]
            annotation = annotations[batch_, :, :]
            two_poinsts_anchor = two_poinsts_anchors[0, :, :]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            ious = batch_cal_iou(two_poinsts_anchor, annotation)
            iou_maxs, iou_max_indices = torch.max(ious, dim=1)

            gt_labels = torch.ones(classification.shape) * -1
            if torch.cuda.is_available():
                gt_labels = gt_labels.cuda()

            # backgroup
            gt_labels[torch.le(iou_maxs, 0.4), :] = 0
            positive_indices = torch.ge(iou_maxs, 0.5)
            # gt
            iou_max_annotation = annotation[iou_max_indices, :]
            gt_labels[positive_indices, :] = 0
            gt_labels[positive_indices, iou_max_annotation[positive_indices, 4].long()] = 1

            # cal classification loss
            alpha_factors = torch.ones(gt_labels.shape) * self.alpha
            if torch.cuda.is_available():
                alpha_factors = alpha_factors.cuda()
            alpha_factors = torch.where(torch.eq(gt_labels, 1.), alpha_factors, 1 - alpha_factors)
            gamma_weights = torch.where(torch.eq(gt_labels, 1.), 1 - classification, classification)
            focal_weights = alpha_factors * torch.pow(gamma_weights, self.gamma)
            bce = -(gt_labels * torch.log(classification) + (1.0 - gt_labels) * torch.log(1.0 - classification))
            classification_loss = focal_weights * bce
            if torch.cuda.is_available():
                classification_loss = torch.where(torch.ne(gt_labels, -1.0), classification_loss,
                                                  torch.zeros(classification_loss.shape).cuda())
            else:
                classification_loss = torch.where(torch.ne(gt_labels, -1.0), classification_loss,
                                                  torch.zeros(classification_loss.shape))

            positive_num = positive_indices.sum()
            classification_loss_mean = classification_loss.sum() / torch.clamp(positive_num.float(), min=1.0)
            classification_losses.append(classification_loss_mean)

            if positive_num > 0:
                positive_regressions = regression[positive_indices, :]
                positive_anchors = anchor[positive_indices, :]
                assigned_gts = iou_max_annotation[positive_indices, :4]

                deta_gts = torch.ones(assigned_gts.shape) * 1.0
                if torch.cuda.is_available():
                    deta_gts = deta_gts.cuda()
                # cal to central format
                deta_gts[:, 2] = assigned_gts[:, 2] - assigned_gts[:, 0]
                deta_gts[:, 0] = assigned_gts[:, 0] + assigned_gts[:, 2] / 2
                deta_gts[:, 3] = assigned_gts[:, 3] - assigned_gts[:, 1]
                deta_gts[:, 1] = assigned_gts[:, 1] + assigned_gts[:, 3] / 2
                # encode gt from anchor
                deta_gts[:, 0] = (deta_gts[:, 0] - positive_anchors[:, 0]) / positive_anchors[:, 2]
                deta_gts[:, 1] = (deta_gts[:, 1] - positive_anchors[:, 1]) / positive_anchors[:, 3]
                deta_gts[:, 2] = torch.log(deta_gts[:, 2] / positive_anchors[:, 2])
                deta_gts[:, 3] = torch.log(deta_gts[:, 3] / positive_anchors[:, 3])
                # smooth L1 loss
                if torch.cuda.is_available():
                    deta_gts = deta_gts / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                    positive_regressions = positive_regressions / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    deta_gts = deta_gts / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                    positive_regressions = positive_regressions / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                regression_losses.append(self.detection_loss(positive_regressions, deta_gts) / torch.clamp(positive_num.float(), 1.0))
                regression_losses.append(self.detection_loss(positive_regressions, deta_gts))
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float())
                else:
                    regression_losses.append(torch.tensor(0).cuda().float())

        return torch.stack(classification_losses).mean(), torch.stack(regression_losses).mean()