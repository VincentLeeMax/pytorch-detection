import argparse

from dataset.voc import VocDataset
from dataset.coco import CocoDataset
from dataset.transform import Normalizer, UniformResizer, ToTensor
from layers.anchor_generate import FPNAnchors
from models.retinanet.retinanet_resnet import RetinaNet

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='voc')
    parser.add_argument('--data_path', help='Path to COCO directory', default='/workspace/nas-data/dataset/voc/VOCdevkit')
    
    dataset = VocDataset('/workspace/nas-data/dataset/voc/VOCdevkit', set_name='VOC2007', 
                       transform=transforms.Compose([Normalizer(), UniformResizer(), ToTensor()]))
    dataloader_train = DataLoader(dataset)
    
    cfg = {'backbone': 'resnet50', 'class_num': 20, 'anchor_num': 9, 'anchor_scales': [2 ** 0, 2 ** (1/3), 2 ** (2/3)], 'anchor_ratios': [0.5, 1, 2]}
    model = RetinaNet(cfg)
    
    anchor_generator = FPNAnchors()
    
    for idx, data in enumerate(dataloader_train):
        classifications, regressions, features = model(data['image'])
        anchors = anchor_generator(data['image'])
    
        print(classifications.shape)
        print(regressions.shape)
        print(anchors.shape)
            
        break
