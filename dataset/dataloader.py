import random

import torch
from torch.utils.data.sampler import Sampler


class AspectRatioBatchSampler(Sampler):
    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.groups = self.get_groups()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def get_groups(self):
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda index: self.dataset.image_aspect_ratio(index))

        groups = [[indices[index % len(indices)] for index in range(i, i + self.batch_size)] for i in
                  range(0, len(indices), self.batch_size)]

        return groups


def padded_collater(batch_data):
    images = [data['image'] for data in batch_data]
    bboxs = [data['bbox'] for data in batch_data]
    scales = [data['scale'] for data in batch_data]

    max_height = max([image.shape[1] for image in images])
    max_width = max([image.shape[2] for image in images])

    padded_image = torch.zeros((len(batch_data), images[0].shape[0], max_height, max_width))
    for idx, image in enumerate(images):
        padded_image[idx, :image.shape[0], :image.shape[1], :image.shape[2]] = image

    max_batch_bbox = max([bbox.shape[0] for bbox in bboxs] + [1])
    padded_bbox = torch.ones((len(batch_data), max_batch_bbox, 5)) * -1
    if max_batch_bbox > 0:
        for idx, bbox in enumerate(bboxs):
            padded_bbox[idx, :bbox.shape[0], :] = bbox


    return {'image': padded_image, 'bbox': padded_bbox, 'scale': scales}


if __name__ == '__main__':
    from voc import VocDataset

    dataset = VocDataset('/data/dataset/VOCdevkit', set_name='VOC2007')
    s = AspectRatioBatchSampler(dataset)
