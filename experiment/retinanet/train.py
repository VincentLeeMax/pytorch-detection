import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import time
import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.voc import VocDataset
from dataset.transform import Normalizer, UniformResizer, ToTensor
from dataset.dataloader import padded_collater, AspectRatioBatchSampler
from layers.anchor_generator import FPNAnchors
from layers.focal_loss import DetectionFocalLoss
from layers.smooth_loss import SmoothL1
from models.retinanet.retinanet_resnet import RetinaNet
from config.cfg import load_cfg
from experiment.retinanet.val import val

if __name__ == '__main__':
    cfg = {'backbone': 'resnet50', 'class_num': 20, 'anchor_num': 9,
           'anchor_scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)],
           'anchor_ratios': [1. / 3, 1, 3]}
    load_cfg(cfg)

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='voc')
    parser.add_argument('--data_path', help='Path to COCO directory', default='/workspace/nas-data/dataset/voc/VOCdevkit')
    args = parser.parse_args()

    dataset_train = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='train',
                       transform=transforms.Compose([Normalizer(), UniformResizer(min_side=300, max_side=600), ToTensor()]))
    dataset_val = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='val',
                         transform=transforms.Compose([Normalizer(), UniformResizer(min_side=300, max_side=600), ToTensor()]))
    cfg['class_num'] = dataset_train.num_classes()
    dataloader_train = DataLoader(dataset_train, num_workers=8, batch_sampler=AspectRatioBatchSampler(dataset_train, batch_size=1),
                                  collate_fn=padded_collater)
    dataloader_val = DataLoader(dataset_val, num_workers=0, batch_sampler=AspectRatioBatchSampler(dataset_val, batch_size=1),
                                  collate_fn=padded_collater)
    anchor_generator = FPNAnchors(anchor_scales=cfg['anchor_scales'], anchor_ratios=cfg['anchor_ratios'])

    result_path = os.path.join('../../result', time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model = RetinaNet(cfg)
    model.freeze_bn()
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    focal_loss = DetectionFocalLoss(detection_loss=SmoothL1())
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    for epoch_num in range(100):
        epoch_loss = []

        # train
        model.train()
        for iter_num, data in enumerate(dataloader_train):
            optimizer.zero_grad()

            annotations = data['bbox']
            images = data['image']
            if torch.cuda.is_available():
                annotations = annotations.cuda()
                images = images.cuda()

            classifications, regressions, features = model(images)
            anchors = anchor_generator(data['image'])
            classification_loss, regression_loss = focal_loss(classifications, regressions, anchors, annotations)

            loss = classification_loss + regression_loss
            loss.backward()

            if loss == 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | learning rate: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, optimizer.param_groups[0]['lr'], float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        print(
            'Epoch: {} | learning rate: {} | Running loss: {:1.5f}'.format(epoch_num,
                                                                           optimizer.param_groups[0]['lr'],
                                                                           np.mean(loss_hist)))

        # val
        val(dataloader_val, anchor_generator, model)

        scheduler.step(np.mean(epoch_loss))
        torch.save(model.modules(), os.path.join(result_path, '{}_retinanet_{}.pt'.format(args.dataset, epoch_num)))