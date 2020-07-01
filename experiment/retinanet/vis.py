import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import cv2
import numpy as np

import torch
from torchvision.ops import nms

from utils.bbox_transform import regress_box, to_twoPoint_format

def vis(input_, anchor_generator, model):
    # val
    model.eval()
    final_scores = []
    final_classes = []
    final_regressed_boxes = []
    final_gt = []

    class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                   'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    transforms = [Normalizer(), UniformResizer(min_side=800, max_side=1000), ToTensor()]
    with torch.no_grad():
        for file_ in os.listdir(input_):
            full_file = os.path.join(input_, file_)
            image = cv2.imread(full_file)
            sample = {'image': image, 'bbox': np.array([[0., 0., 0., 0., 0.]])}
            for transform in transforms:
                sample = transform(sample)
            image_ = sample['image'].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(input_, 'aug_' + file_), image_)
            image = UniformResizer(min_side=800, max_side=1000)({'image': image, 'bbox': np.array([[0., 0., 0., 0., 0.]])})['image']
            images = sample['image']
            if torch.cuda.is_available():
                images = images.cuda()

            classifications, regressions, _ = model(images.unsqueeze(dim=0))
            anchors = anchor_generator(sample['image'])

            regressed_boxes = regress_box(anchors, regressions)
            regressed_boxes = to_twoPoint_format(regressed_boxes)

            # filter some lowest case
            num_class = classifications.shape[-1]

            # batch
            max_scores = torch.max(classifications[0], dim=1)[0]
            max_indices = max_scores > 0.4

            if torch.sum(max_indices) > 0:
                selected_scores = max_scores[max_indices]
                selected_classifications = classifications[0, max_indices, :].reshape(-1, num_class)
                selected_regressed_boxes = regressed_boxes[0, max_indices, :].reshape(-1, 4)

                keep_dim = nms(selected_regressed_boxes, selected_scores, iou_threshold=0.5)
                final_classifications = selected_classifications[keep_dim, :]
                maxs = torch.max(final_classifications, dim=1)
                final_scores = maxs[0]
                final_classes = maxs[1]
                final_regressed_boxes = selected_regressed_boxes[keep_dim, :]


                for idx, _ in enumerate(final_scores):
                    cv2.rectangle(image,
                                  (int(final_regressed_boxes[idx][0].item()), int(final_regressed_boxes[idx][1].item())),
                                  (int(final_regressed_boxes[idx][2].item()), int(final_regressed_boxes[idx][3].item())),
                                  126, 2)
                    cv2.putText(image,
                                    class_names[final_classes[idx].item()] + '_' + str(final_scores[idx].item()),
                                    (int(final_regressed_boxes[idx][0]), int(final_regressed_boxes[idx][1] + 4)),
                                    cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 1)

            else:
                final_scores = []
                final_classes = []
                final_regressed_boxes = []

            cv2.imwrite(os.path.join(input_, 'result_' + file_), image)

            print(final_scores)


    return final_scores, final_classes, final_regressed_boxes, final_gt



if __name__ == '__main__':
    from layers.anchor_generator import FPNAnchors
    from dataset.transform import Normalizer, UniformResizer, ToTensor

    cfg = {'backbone': 'resnet50', 'class_num': 20, 'anchor_num': 9, 'anchor_scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)], 'anchor_ratios': [1. / 3, 1, 3]}
    anchor_generator = FPNAnchors(anchor_scales=cfg['anchor_scales'], anchor_ratios=cfg['anchor_ratios'])

    model = torch.load('/data/workspace/pytorch-detection/result/voc_retinanet_60.pt')
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    final_scores, final_classes, final_regressed_boxes, final_gt = vis('/data/dataset/VOCdevkit/VOC2007/test', anchor_generator, model)