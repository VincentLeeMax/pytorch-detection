import numpy as np

import torch

from utils.iou import batch_cal_iou

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calculate_voc_map(classes, det_scores, det_classes, dets, gts, iou_thresh=0.5, use_07_metric=False):
    """
    classes: class label
    det_scores: [N, 1]
    det_classes: [N, 1]
    dets: [N, 4]
    gts:  [N, 5]
    """
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    dets_dict = {}

    gts_dict = {}
    for idx_class, class_ in enumerate(classes):
        dets_dict.setdefault(idx_class, {})
        dets_dict[idx_class].setdefault('scores', [])
        dets_dict[idx_class].setdefault('classes', [])
        dets_dict[idx_class].setdefault('bbox', [])
        dets_dict[idx_class].setdefault('idx', [])

        gts_dict.setdefault(idx_class, {})
        for idx_, _ in enumerate(dets):
            idx_dets = dets[idx_]
            idx_det_scores = det_scores[idx_]
            idx_det_classes = det_classes[idx_]
            idx_gts = gts[idx_]

            det_keep = torch.where(idx_det_classes[:] == idx_class)[0]
            gt_keep = torch.where(idx_gts[:, 4] == idx_class)[0]

            gts_dict[idx_class].setdefault(idx_, {})
            gts_dict[idx_class][idx_]['bbox'] = idx_gts[gt_keep, :]
            gts_dict[idx_class][idx_]['selected'] = torch.zeros((gt_keep.shape[0], 1))
            gts_dict[idx_class].setdefault('num', 0)
            gts_dict[idx_class]['num'] += gt_keep.shape[0]


            for k in det_keep:
                dets_dict[idx_class]['scores'].append(idx_det_scores[k.item()].item())
                dets_dict[idx_class]['classes'].append(idx_det_classes[k.item()].item())
                dets_dict[idx_class]['bbox'].append(idx_dets[k.item()])
                dets_dict[idx_class]['idx'].append(idx_)

    map = []
    for idx_class, class_ in enumerate(classes):
        if class_ == '__background__':
            continue

        orders_ = np.argsort(-np.array(dets_dict[idx_class]['scores']))
        tp = np.zeros(len(orders_))
        fp = np.zeros(len(orders_))
        for idx_order_, order_ in enumerate(orders_):
            idx_ = dets_dict[idx_class]['idx'][order_]
            gts = gts_dict[idx_class][idx_]['bbox']
            dets = dets_dict[idx_class]['bbox'][order_]
            if dets.shape[0] == 0 or gts.shape[0] == 0:
                continue
            ious = batch_cal_iou(dets.unsqueeze(dim=0), gts)
            iou_maxs, iou_max_indices = torch.max(ious, dim=1)
            if iou_maxs[0] > iou_thresh and gts_dict[idx_class][idx_]['selected'][iou_max_indices[0]] == 0:
                tp[idx_order_] = 1
                gts_dict[idx_class][idx_]['selected'][iou_max_indices[0]] = 1
            else:
                fp[idx_order_] = 1

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(gts_dict[idx_class]['num'])
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        map.append(ap)

        print('AP for {} = {:.4f}'.format(class_, ap))
    print('Mean AP = {:.4f}'.format(np.mean(map)))
    print('~~~~~~~~')

if __name__ == '__main__':
    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    det_scores = []
    det_classes = []
    dets = []
    gts = []
    for idx in range(5):
        det_scores.append(torch.rand((2, 1)))
        det_classes.append(torch.randint(0, 20, (2, 1)))
        dets.append(torch.randint(0, 100, (2, 4)))

        gts.append(torch.randint(0, 100, (2, 5)))
        gts[idx][-1] = torch.randint(0, 20, (1, 1))

    print(det_scores)
    print(det_classes)
    print(dets)
    print(gts)

    calculate_voc_map(class_names, det_scores, det_classes, dets, gts)
