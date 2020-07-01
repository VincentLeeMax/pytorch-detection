import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.vgg import VGGModels as VGG
from layers.anchor_generator import SSDAnchors

class PredictHead(nn.Module):
    def __init__(self, num_class, boxes_per_locations, out_channels):
        super(PredictHead, self).__init__()
        self.num_class = num_class
        self.boxes_per_locations = boxes_per_locations
        self.out_channels = out_channels
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        self.init_layer()
        self._initialize_weights()

    def init_layer(self):
        for boxes_per_location, out_channel in zip(self.boxes_per_locations, self.out_channels):
            self.cls_headers.append(
                nn.Conv2d(out_channel, boxes_per_location * self.num_class, kernel_size=3, stride=1, padding=1))
            self.reg_headers.append(
                nn.Conv2d(out_channel, boxes_per_location * 4, kernel_size=3, stride=1, padding=1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        class_head_result = []
        reg_head_result = []
        for feature, cls_header, reg_header in zip(features, self.cls_headers, self.reg_headers):
            class_head_result.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            reg_head_result.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())

        batch_size = features[0].shape[0]
        classfication = torch.cat([result.view(result.shape[0], -1) for result in class_head_result], dim=1).view(batch_size, -1, self.num_class)
        regression = torch.cat([result.view(result.shape[0], -1) for result in reg_head_result], dim=1).view(batch_size, -1, 4)

        return classfication, regression


class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.training = True
        self.arch = cfg['backbone']
        self.class_num = cfg['class_num']
        self.boxes_per_locations = cfg['box_per_locations']
        self.anchor_box_generator = SSDAnchors(level_strides=cfg['strides'],
                       anchor_sizes=cfg['anchor_sizes'],
                       anchor_ratios=cfg['anchor_ratios'],
                       feature_sizes=cfg['feature_maps'])
        self.anchor_boxs = None
        self.extractor = VGG[self.arch](pretrained=True, progress=True)
        self.predictor = PredictHead(self.class_num, self.boxes_per_locations, self.extractor.get_out_channels())

    def forward(self, images):
        features = self.extractor(images)
        classfication, regression = self.predictor(features)
        if self.anchor_boxs is None:
            self.anchor_boxs = self.anchor_box_generator(images)
        if self.training:
            return classfication, regression, self.anchor_boxs
        else:
            classfication = F.softmax(classfication, dim=2)

            return classfication, regression, self.anchor_boxs