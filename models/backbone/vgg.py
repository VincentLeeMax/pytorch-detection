import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.l2Norm import L2Norm

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    'VGGModels',
]


model_urls = {
    # 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm):
        super(VGG, self).__init__()
        self.batch_norm = batch_norm
        self.features, self.conv4_3_index = vgg_base(cfgs[cfg][0], batch_norm)
        self.features_extra = vgg_extras(cfgs[cfg][1], cfg)
        self._initialize_weights()

        if not batch_norm:
            self.l2_norm = L2Norm(512, scale=20)

    def forward(self, x):
        features = []
        for feature in self.features[:self.conv4_3_index]:
            x = feature(x)
        if self.l2_norm is not None:
            s = self.l2_norm(x)
        features.append(s)
        for feature in self.features[self.conv4_3_index:]:
            x = feature(x)
        features.append(x)
        for k, v in enumerate(self.features_extra):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return features

    def init_from_pretrain(self, state_dict):
        self.features.load_state_dict(state_dict)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def get_out_channels(self):
        out_channels = []
        if self.batch_norm:
            out_channels += [self.features[self.conv4_3_index - 3].out_channels, self.features[-3].out_channels]
        else:
            out_channels += [self.features[self.conv4_3_index - 2].out_channels, self.features[-2].out_channels]
        for k, v in enumerate(self.features_extra):
            if k % 2 == 1:
                out_channels.append(v.out_channels)

        return out_channels


def vgg_base(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    conv4_3_index = 0
    conv_num = 1
    for v in cfg:
        if v == 'M':
            conv_num += 1
            layer = [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            conv_num += 1
            layer = [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layer = [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer = [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        if conv_num < 5:
            conv4_3_index += len(layer)
        layers += layer

    # extend layer
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(in_channels, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    if batch_norm:
        layers += [pool5, conv6, nn.BatchNorm2d(1024),
                   nn.ReLU(inplace=True), conv7, nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    else:
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]

    return nn.ModuleList(layers), conv4_3_index


def vgg_extras(cfg, cfg_size):
    layers = []
    stride2 = False
    in_channel = 1024
    for k, v in enumerate(cfg):
        if in_channel != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channel, cfg[k+1], kernel_size=(1, 3)[stride2], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channel, v, kernel_size=(1, 3)[stride2])]

            stride2 = not stride2
        in_channel = v
    if '512' in cfg_size:
        layers.append(nn.Conv2d(in_channel, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))

    return nn.Sequential(*layers)


cfgs = {
    'vgg16_300': [
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    ],
    'vgg16_512': [
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
        [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
    ]
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = VGG(cfg, batch_norm, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              model_dir='./pretrain')
        model.init_from_pretrain(state_dict)
        # model.load_state_dict(state_dict, strict=False)

    return model


def vgg16_300(pretrained=True, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'vgg16_300', False, pretrained, progress, **kwargs)


def vgg16_512(pretrained=True, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'vgg16_512', False, pretrained, progress, **kwargs)

VGGModels = {'vgg16_300': vgg16_300, 'vgg16_512': vgg16_512}