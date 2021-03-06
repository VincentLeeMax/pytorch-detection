import cv2
import numpy as np

import torch 

class Normalizer(object):
    """Normalize with mean and std"""
    def __init__(self):
        self.mean = np.array([[[123, 117, 104]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['image'], sample['bbox']

        # return {'image':((image.astype(np.float32)-self.mean)/self.std), 'bbox': annots}
        return {'image': (image.astype(np.float32) - self.mean), 'bbox': annots}
    
class UniformResizer(object):
    """Resize image and annotation"""
    def __init__(self, min_side=800, max_side=1000):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image,(int(round((cols*scale))),int(round(rows*scale))), interpolation=cv2.INTER_CUBIC)
        
        bbox[:, :4] *= scale

        return {'image': image, 'bbox': bbox, 'scale': [scale, scale]}


class Resizer(object):
    """Resize image and annotation"""

    def __init__(self, size=512):
        self.size = size

    def __call__(self, sample):
        image, bbox = sample['image'], sample['bbox']

        rows, cols, cns = image.shape

        # resize the image with the computed scale
        image = cv2.resize(image, (self.size, self.size))

        scale = [self.size * 1.0 / cols,  self.size * 1.0 / rows]
        bbox[:, 0] *= scale[0]
        bbox[:, 1] *= scale[1]
        bbox[:, 2] *= scale[0]
        bbox[:, 3] *= scale[1]

        return {'image': image, 'bbox': bbox, 'scale': scale}

    
class ToTensor(object):
    """Change ndarrays to tensor"""
    def __call__(self, sample):
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1).unsqueeze(0).float()
        sample['bbox'] = torch.from_numpy(sample['bbox']).float()

        if torch.cuda.is_available():
            sample['image'] = sample['image'].cuda()
            sample['bbox'] = sample['bbox'].cuda()
        
        return sample