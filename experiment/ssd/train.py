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
from layers.box_loss import MultiBoxLoss
from dataset.collator import BatchCollator
from solver.lr_scheduler import WarmupMultiStepLR
from utils.metricLogger import MetricLogger
from experiment.ssd.val import val

if __name__ == '__main__':
    cfg = {'backbone': 'vgg16_300',
           'feature_maps': [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
           'strides': [8, 16, 32, 64, 100, 300],
           'anchor_sizes': [[30, math.sqrt(30*60), 30, 30],
                             [60, math.sqrt(60*111), 60, 60, 60, 60],
                             [111, math.sqrt(111*162), 111, 111, 111, 111],
                             [162, math.sqrt(162*213), 162, 162, 162, 162],
                             [213, math.sqrt(213*264), 213, 213],
                             [264, math.sqrt(264*315), 264, 264]],
           'anchor_ratios': [[1, 1, 0.5, 2],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2, 1. / 3, 3],
                             [1, 1, 0.5, 2],
                             [1, 1, 0.5, 2]],
           'box_per_locations': [4, 6, 6, 6, 4, 4]}
    load_cfg(cfg)

    result_path = os.path.join('result', time.strftime("%Y%m%d%H%M%S", time.localtime()))
    # result_path = 'result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dataset_train = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='trainval',
                               transform=transforms.Compose([Resizer(size=300), Normalizer(), ToTensor()]))
    dataset_val = VocDataset('/data/dataset/VOCdevkit/VOC2007', set_name='trainval',
                             transform=transforms.Compose([Resizer(size=300), Normalizer(), ToTensor()]))
    cfg['class_num'] = dataset_train.num_classes()
    dataloader_train = DataLoader(dataset_train, batch_size=8, num_workers=0, collate_fn=BatchCollator())
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=0, collate_fn=BatchCollator())

    model = SSD(cfg)
    if torch.cuda.is_available():
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()
    # else:
        # model = torch.nn.DataParallel(model)

    multiBoxLoss = MultiBoxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = WarmupMultiStepLR(optimizer=optimizer,
                                  milestones=[120 * len(dataloader_train), 160 * len(dataloader_train)],
                                  gamma=0.1,
                                  warmup_factor=1.0 / 3,
                                  warmup_iters=500)

    metricLogger = MetricLogger()
    for epoch_num in range(200):
        loss_hist = []
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            model.train()
            optimizer.zero_grad()
            gt_boxes = data['bbox']
            images = data['image']

            classification, regression, anchors = model(images)
            loss_dict = multiBoxLoss(classification, regression, anchors, gt_boxes)
            loss = sum(loss_dict.values())

            loss.backward()
            optimizer.step()
            scheduler.step()

            metricLogger.update(total_loss=loss, **loss_dict)
            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            if iter_num % 10 == 0:
                print(
                    'Epoch: {} | Iteration: {} | learning rate: {:1.7f} | {}'.format(
                        epoch_num, iter_num, optimizer.param_groups[0]['lr'], str(metricLogger)))

            del loss


        print(
            'Epoch: {} | learning rate: {:1.5f} | Running loss: {:1.7f}'.format(epoch_num,
                                                                           optimizer.param_groups[0]['lr'],
                                                                           np.mean(epoch_loss)))

        if epoch_num % 10 == 0:
            torch.save(model.state_dict(), os.path.join(result_path, '{}_vgg_{}.pt'.format(cfg['backbone'], epoch_num)))

            # val(dataloader_val, model)