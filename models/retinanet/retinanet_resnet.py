import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from backbone.resnet import ResNetModels as ResNet

class FeaturePyramid(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, channel_=256):
        super(FeaturePyramid, self).__init__()
        
        self.C5_reduction = nn.Conv2d(C5_size, channel_, kernel_size=1, stride=1, padding=0)
        self.P5_conv = nn.Conv2d(channel_, channel_, kernel_size=3, stride=1, padding=1)
        
        self.C4_reduction = nn.Conv2d(C4_size, channel_, kernel_size=1, stride=1, padding=0)
        self.P4_conv = nn.Conv2d(channel_, channel_, kernel_size=3, stride=1, padding=1)
        
        self.C3_reduction = nn.Conv2d(C3_size, channel_, kernel_size=1, stride=1, padding=0)
        self.P3_conv = nn.Conv2d(channel_, channel_, kernel_size=3, stride=1, padding=1)
        
        self.P6_conv = nn.Conv2d(channel_, channel_, kernel_size=3, stride=2, padding=1)
        
        self.P7_conv = nn.Conv2d(channel_, channel_, kernel_size=3, stride=2, padding=1)
        self.P7_act = nn.ReLU()
    
    def _arbitrary_upsample(self, feature_, refer_):
        
        return F.interpolate(feature_, size=(refer_.shape[-2], refer_.shape[-1]), mode='bilinear', align_corners=False)
    
    def forward(self, x):
        C3, C4, C5 = x
        
        P5_r = self.C5_reduction(C5)
        P5_c = self.P5_conv(P5_r)
        P5_u = self._arbitrary_upsample(P5_r, C4)
        
        P4_r = self.C4_reduction(C4) + P5_u
        P4_c = self.P4_conv(P4_r)
        P4_u = self._arbitrary_upsample(P4_r, C3)
        
        P3_r = self.C3_reduction(C3) + P4_u
        P3_c = self.P3_conv(P3_r)
        
        P6_c = self.P6_conv(P5_r)
        
        P7_c = self.P7_conv(P6_c)
        P7_c = self.P7_act(P7_c)
        
        return [P3_c, P4_c, P5_c, P6_c, P7_c]

class ClassificationSubnet(nn.Module):
    def __init__(self, class_num, anchor_num, channel_in=256):
        super(ClassificationSubnet, self).__init__()
        self.class_num = class_num
        self.anchor_num = anchor_num
        
        self.conv1 = nn.Conv2d(channel_in, 256, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        
        self.conv_output = nn.Conv2d(256, class_num * anchor_num, kernel_size=3, stride=1, padding=1)
        self.act_output = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.act3(out)
        
        out = self.conv4(out)
        out = self.act4(out)
        
        out = self.conv_output(out)
        out = self.act_output(out)
        
        batch_size = out.shape[0]
        # out is B x C x H x W, out_permute is B x H x W x C
        out_permute = out.permute(0, 2, 3, 1)
        
        # easy to softmax
        out_reshape = out_permute.contiguous().view(batch_size, -1, self.class_num)
        
        return out_reshape
    
class RegressionSubnet(nn.Module):
    def __init__(self, anchor_num, channel_in=256):
        super(RegressionSubnet, self).__init__()
        self.anchor_num = anchor_num
        
        self.conv1 = nn.Conv2d(channel_in, 256, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.act4 = nn.ReLU()
        
        self.conv_output = nn.Conv2d(256, 4 * anchor_num, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.act2(out)
        
        out = self.conv3(out)
        out = self.act3(out)
        
        out = self.conv4(out)
        out = self.act4(out)
        
        out = self.conv_output(out)
        batch_size = out.shape[0]
        # out is B x C x H x W, out_permute is B x H x W x C
        out_permute = out.permute(0, 2, 3, 1)
        
        # easy to softmax
        out_reshape = out_permute.contiguous().view(batch_size, -1, 4)
        
        return out_reshape
        

class RetinaNet(nn.Module):
    def __init__(self, cfg):
        super(RetinaNet, self).__init__()
        self.arch = cfg['backbone']
        self.class_num = cfg['class_num']
        self.anchor_num = cfg['anchor_num']
        
        self.extractor = ResNet[cfg['backbone']](pretrained=False, progress=True, return_feature=True)
        feature_sizes = self.extractor.get_feature_size()
        self.fpn = FeaturePyramid(feature_sizes[0], feature_sizes[1], feature_sizes[2])
        self.classificationSubnet = ClassificationSubnet(self.class_num, self.anchor_num)
        self.regressionSubnet = RegressionSubnet(self.anchor_num)
        
        prior = 0.01
        
        self.classificationSubnet.conv_output.weight.data.fill_(0)
        self.classificationSubnet.conv_output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionSubnet.conv_output.weight.data.fill_(0)
        self.regressionSubnet.conv_output.bias.data.fill_(0)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.extractor.load_weights(self.arch)
                
    def forward(self, x):
        C3, C4, C5 = self.extractor(x)
        features = self.fpn([C3, C4, C5])
        
        classifications = torch.cat([self.classificationSubnet(feature) for feature in features], dim=1)
        regressions = torch.cat([self.regressionSubnet(feature) for feature in features], dim=1)
        
        
        return [classifications, regressions, features]