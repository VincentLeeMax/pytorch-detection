import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.voc import VocDataset
from dataset.coco import CocoDataset
from dataset.transform import Normalizer, UniformResizer, ToTensor
from dataset.dataloader import padded_collater, AspectRatioBatchSampler
from layers.anchor_generate import FPNAnchors
from layers.focal_loss import DetectionFocalLoss
from layers.smooth_loss import SmoothL1
from models.retinanet.retinanet_resnet import RetinaNet
from config.cfg import C_, load_cfg

if __name__ == '__main__':
    cfg = {'backbone': 'resnet50', 'class_num': 20, 'anchor_num': 9, 'anchor_scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)], 'anchor_ratios': [1. / 3, 1, 3]}
    load_cfg(cfg)

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='voc')
    parser.add_argument('--data_path', help='Path to COCO directory', default='/workspace/nas-data/dataset/voc/VOCdevkit')

    dataset = VocDataset('/data/dataset/VOCdevkit', set_name='VOC2007',
                       transform=transforms.Compose([Normalizer(), UniformResizer(), ToTensor()]))
    # dataset = CocoDataset('/data/dataset/coco', set_name='train2017',
    #                      transform=transforms.Compose([Normalizer(), UniformResizer(), ToTensor()]))
    cfg['class_num'] = dataset.num_classes()
    aspect_sampler = AspectRatioBatchSampler(dataset, batch_size=2)
    dataloader_train = DataLoader(dataset, num_workers=1, batch_sampler=aspect_sampler, collate_fn=padded_collater)
    anchor_generator = FPNAnchors(anchor_scales=cfg['anchor_scales'], anchor_ratios=cfg['anchor_ratios'])

    model = RetinaNet(cfg)
    model.freeze_bn()
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    focal_loss = DetectionFocalLoss(detection_loss=SmoothL1(reduction='mean'))
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)
    for epoch_num in range(100):
        epoch_loss = []

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
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        scheduler.step(np.mean(epoch_loss))
