import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torchvision.ops import nms

from utils.bbox_transform import regress_box, to_cornor_form
from utils.map import calculate_voc_map

def val(dataloader_val, model):
    # val
    model.eval()
    final_scores = []
    final_classes = []
    final_regressed_boxes = []
    final_gt = []

    with torch.no_grad():
        for iter_num, data in enumerate(dataloader_val):
            annotations = data['bbox'][0]
            images = data['image']

            if torch.cuda.is_available():
                annotations = annotations.cuda()
                images = images.cuda()

            classifications, regressions, anchors = model(images)

            regressions = regressions * torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
            regressed_boxes = regress_box(anchors, regressions)
            regressed_boxes = to_cornor_form(regressed_boxes)

            # filter some lowest case
            batch_size = classifications.shape[0]
            num_class = classifications.shape[-1]

            assert batch_size == 1

            max_scores = torch.max(classifications[0, :, 1:], dim=1)[0]
            max_indices = max_scores > 0.01
            selected_scores = max_scores[max_indices]
            selected_classifications = classifications[0][max_indices, :].reshape(-1, num_class)
            selected_regressed_boxes = regressed_boxes[0][max_indices, :].reshape(-1, 4)

            keep_dim = nms(selected_regressed_boxes, selected_scores, iou_threshold=0.5)
            final_classifications = selected_classifications[keep_dim, :]
            if final_classifications.shape[0] == 0:
                final_scores.append(torch.zeros(0, 1))
                final_classes.append(torch.zeros(0, 1))
                final_regressed_boxes.append(torch.zeros(0, 4))
            else:
                maxs = torch.max(final_classifications, dim=1)
                final_scores.append(maxs[0])
                final_classes.append(maxs[1])
                final_regressed_boxes.append(selected_regressed_boxes[keep_dim, :])

            final_gt.append(annotations)

            if iter_num % 100 == 0:
                print("{}/{}".format(iter_num, len(dataloader_val)))

    calculate_voc_map(dataloader_val.dataset.class_names, final_scores, final_classes, final_regressed_boxes, final_gt)

    return final_scores, final_classes, final_regressed_boxes, final_gt



if __name__ == '__main__':
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import math
    import numpy as np
    import time

    import torch
    from torchvision import transforms
    from torch.utils.data import DataLoader

    from config.cfg import load_cfg
    from dataset.voc import VocDataset
    from models.ssd.ssd_vgg import SSD
    from dataset.transform import Normalizer, Resizer, ToTensor
    from dataset.collator import BatchCollator

    if __name__ == '__main__':
        cfg = {'backbone': 'vgg16_300',
               'feature_maps': [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
               'strides': [8, 16, 32, 64, 100, 300],
               'anchor_sizes': [[30, math.sqrt(30 * 60), 30, 30],
                                [60, math.sqrt(60 * 111), 60, 60, 60, 60],
                                [111, math.sqrt(111 * 162), 111, 111, 111, 111],
                                [162, math.sqrt(162 * 213), 162, 162, 162, 162],
                                [213, math.sqrt(213 * 264), 213, 213],
                                [264, math.sqrt(264 * 315), 264, 264]],
               'anchor_ratios': [[1, 1, 0.5, 2],
                                 [1, 1, 0.5, 2, 1. / 3, 3],
                                 [1, 1, 0.5, 2, 1. / 3, 3],
                                 [1, 1, 0.5, 2, 1. / 3, 3],
                                 [1, 1, 0.5, 2],
                                 [1, 1, 0.5, 2]],
               'box_per_locations': [4, 6, 6, 6, 4, 4],
               'class_num': 21}
        load_cfg(cfg)

    dataset_val = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='test',
                             transform=transforms.Compose(
                                 [Normalizer(), Resizer(size=300), ToTensor()]))
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=0, collate_fn=BatchCollator())

    model = SSD(cfg)
    model.load_state_dict(torch.load('/data/workspace/pytorch_learning/pytorch-detection/result/20200627145838/vgg16_300_vgg_190.pt'))
    import time
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    final_scores, final_classes, final_regressed_boxes, final_gt = val(dataloader_val, model)