import os
import random
import urllib.request
import zipfile
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class CocoDataset(Dataset):

    def __init__(self, root_dir, set_name='train2017', transform=None, need_background=True):
        self.need_background = need_background
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        annotation_path = os.path.join(
            self.root_dir,
            'annotations',
            'instances_' + self.set_name + '.json'
        )
        self.coco = COCO(annotation_path)
        self.image_ids = self.coco.getImgIds()
        self.image_ids = [id for id in self.image_ids if len(self.coco.getAnnIds(imgIds=id, iscrowd=False)) > 0]

        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            id = c['id']
            if self.need_background:
                id += 1
            self.coco_labels[len(self.classes)] = id
            self.coco_labels_inverse[id] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.class_names = list(self.classes.keys())
        if self.need_background:
            self.class_names = ['background'] + self.class_names

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        bbox = self.load_annotations(idx)

        sample = {'image': image, 'bbox': bbox}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = cv2.imread(path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def classes(self):
        return self.class_names

    def num_classes(self):
        return len(self.class_names)

if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from dataset.transform import Normalizer, UniformResizer, ToTensor

    coco = CocoDataset('/workspace/nas-data/dataset/coco', set_name='val2014',
                       transform=transforms.Compose([Normalizer(), UniformResizer(), ToTensor()]))
    dataloader_train = DataLoader(coco)
    print(coco.labels)
    print(coco.classes)
    print(coco.coco_labels)
    print(coco.coco_labels_inverse)
    print(coco.image_ids[:10])
    for iter_num, data in enumerate(dataloader_train):
        print(data['image'].shape)
        break
