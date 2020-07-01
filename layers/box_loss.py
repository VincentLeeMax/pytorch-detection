import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.mining import hard_negative_mining
from utils.iou import batch_cal_iou
from utils.bbox_transform import regress_box, to_cornor_form, to_central_form

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, classifications, regressions, anchors, annotations):
        classification_losses = []
        regression_losses = []

        two_poinsts_anchors = to_cornor_form(anchors)
        for batch_ in range(classifications.shape[0]):
            classification = classifications[batch_, :, :]
            regression = regressions[batch_, :, :]
            anchor = anchors[0, :, :]
            annotation = annotations[batch_]
            two_poinsts_anchor = two_poinsts_anchors[0, :, :]

            if annotation.shape[0] < 1:
                if torch.cuda.is_available():
                    classification_losses.append(torch.tensor(0).cuda())
                    regression_losses.append(torch.tensor(0).cuda())
                else:
                    classification_losses.append(torch.tensor(0))
                    regression_losses.append(torch.tensor(0))
                continue

            # classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            ious = batch_cal_iou(two_poinsts_anchor, annotation)
            # make sure every target have at least one anchor
            _, target_iou_max_indices = torch.max(ious, dim=0)
            for idx, target_iou_max_indice in enumerate(target_iou_max_indices):
                ious[target_iou_max_indice, idx] = 2
            # find max target for every anchor
            iou_maxs, iou_max_indices = torch.max(ious, dim=1)

            gt_labels = torch.zeros((classification.shape[0]))
            if torch.cuda.is_available():
                gt_labels = gt_labels.cuda()

            # gt
            positive_indices = torch.ge(iou_maxs, 0.5)
            iou_max_annotation = annotation[iou_max_indices, :]
            gt_labels[positive_indices] = iou_max_annotation[positive_indices, 4]

            num_classes = classification.size(1)
            with torch.no_grad():
                loss = -F.log_softmax(classification, dim=1)[:, 0]
                mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

            positive_num = positive_indices.sum()
            classification = classification[mask, :].view(-1, num_classes)
            classification_loss = F.cross_entropy(classification, gt_labels[mask].long(), reduction='sum')
            classification_losses.append(classification_loss / positive_num)

            positive_regressions = regression[positive_indices, :]
            positive_anchors = anchor[positive_indices, :]
            assigned_gts = iou_max_annotation[positive_indices, :4]

            # cal to central format
            deta_gts = to_central_form(assigned_gts.unsqueeze(0))[0]

            # encode gt from anchor
            deta_gts[:, 0] = (deta_gts[:, 0] - positive_anchors[:, 0]) / positive_anchors[:, 2]
            deta_gts[:, 1] = (deta_gts[:, 1] - positive_anchors[:, 1]) / positive_anchors[:, 3]
            deta_gts[:, 2] = torch.log(deta_gts[:, 2] / positive_anchors[:, 2])
            deta_gts[:, 3] = torch.log(deta_gts[:, 3] / positive_anchors[:, 3])


            # smooth L1 loss
            if torch.cuda.is_available():
                deta_gts = deta_gts / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                # positive_regressions = positive_regressions / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
            else:
                deta_gts = deta_gts / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
                # positive_regressions = positive_regressions / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])
            smooth_l1_loss = F.smooth_l1_loss(positive_regressions, deta_gts, reduction='sum')

            regression_losses.append(smooth_l1_loss / positive_num)

            # regressions = positive_regressions.unsqueeze(0) * torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
            # regressed_boxes = regress_box(positive_anchors.unsqueeze(0), regressions)
            # regressed_boxes = to_cornor_form(regressed_boxes)
            # print(regressed_boxes)
            # print(annotation.unsqueeze(0))

        return {'cls_loss': torch.stack(classification_losses).mean(), 'reg_loss': torch.stack(regression_losses).mean()}