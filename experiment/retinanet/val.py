import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torchvision.ops import nms

from utils.bbox_transform import regress_box, to_twoPoint_format
from utils.map import calculate_voc_map

def val(dataloader_val, anchor_generator, model):
    # val
    model.eval()
    final_scores = []
    final_classes = []
    final_regressed_boxes = []
    final_gt = []

    with torch.no_grad():
        for iter_num, data in enumerate(dataloader_val):
            annotations = data['bbox']
            images = data['image']

            if torch.cuda.is_available():
                annotations = annotations.cuda()
                images = images.cuda()

            classifications, regressions, _ = model(images)
            anchors = anchor_generator(data['image'])

            regressed_boxes = regress_box(anchors, regressions)
            regressed_boxes = to_twoPoint_format(regressed_boxes)

            # filter some lowest case
            batch_size = classifications.shape[0]
            num_class = classifications.shape[-1]

            # batch
            for idx_ in range(batch_size):
                max_scores = torch.max(classifications[idx_], dim=1)[0]
                max_indices = max_scores > 0.05
                selected_scores = max_scores[max_indices]
                selected_classifications = classifications[idx_][max_indices, :].reshape(-1, num_class)
                selected_regressed_boxes = regressed_boxes[idx_][max_indices, :].reshape(-1, 4)

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
                final_gt.append(annotations[idx_])

    calculate_voc_map(dataloader_val.dataset.class_names, final_scores, final_classes, final_regressed_boxes, final_gt)

    return final_scores, final_classes, final_regressed_boxes, final_gt



if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms

    from dataset.voc import VocDataset
    from layers.anchor_generator import FPNAnchors
    from dataset.transform import Normalizer, UniformResizer, ToTensor
    from dataset.dataloader import padded_collater, AspectRatioBatchSampler
    from utils.map import calculate_voc_map

    cfg = {'backbone': 'resnet50', 'class_num': 20, 'anchor_num': 9, 'anchor_scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)], 'anchor_ratios': [1. / 3, 1, 3]}

    dataset_val = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='val2',
                             transform=transforms.Compose(
                                 [Normalizer(), UniformResizer(min_side=300, max_side=600), ToTensor()]))
    dataloader_val = DataLoader(dataset_val, num_workers=1,
                                batch_sampler=AspectRatioBatchSampler(dataset_val, batch_size=2),
                                collate_fn=padded_collater)
    anchor_generator = FPNAnchors(anchor_scales=cfg['anchor_scales'], anchor_ratios=cfg['anchor_ratios'])

    model = torch.load('/data/workspace/pytorch-detection/result/20200421082134/voc_retinanet_2.pt')
    import time
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    final_scores, final_classes, final_regressed_boxes, final_gt = val(dataloader_val, anchor_generator, model)