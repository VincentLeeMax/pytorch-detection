import torch
import torch.nn as nn

from backbone.vgg import VGGModels as VGG

class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__()
        self.extractor = VGG[cfg['BACKBONE']](True, True)
        
        
    def forward(self, x):
        x = self.extractor(x)
        
        
        return x