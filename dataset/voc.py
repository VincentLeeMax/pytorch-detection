import os
import random
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import pickle

import torch
from torch.utils.data import Dataset

def load_xml_annotaitons(path, class_to_ind, keep_difficult=False):
    root = ET.parse(path).getroot()
    res = []
    
    for obj in root.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if not keep_difficult and difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')

        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            # scale height or width
            bndbox.append(cur_pt)
        label_idx = class_to_ind[name]
        bndbox.append(label_idx)
        res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
        # img_id = target.find('filename').text[:-4]
    
    # [[xmin, ymin, xmax, ymax, label_ind], ... ]
    
    return np.array(res, dtype=np.float64)

def load_xml_meta(path):
    print(path)
    root = ET.parse(path).getroot()
    
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    
    return {"height":height, "width":width}

class VocDataset(Dataset):

    def __init__(self, root_dir, set_name='VOC2007', transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.load_image_ids()
        self.load_image_infos()
        self.load_classes()

    def load_classes(self):
        categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                      'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                      'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.voc_labels         = {}
        self.voc_labels_inverse = {}
        for idx, c in enumerate(categories):
            self.voc_labels[c] = idx + 1
            self.voc_labels_inverse[idx + 1] = c
        
    def load_image_ids(self):
        self.image_ids = []
        path_ = os.path.join(self.root_dir, self.set_name, 'Annotations')
        
        for file_ in os.listdir(path_):
            postFix = '.' + file_.split('.')[-1]
            image_id = file_.replace(postFix, '')
            self.image_ids.append(image_id)
    
    def load_image_infos(self):
        print('loading meta annotations...')
        
        self.image_infos = {}
        path_ = os.path.join('.', 'cache', self.set_name)
        if os.path.exists(path_):
            cache_ = open(path_, 'rb')
            self.image_infos = pickle.load(cache_)
            cache_.close()
        else:
            for image_id in self.image_ids:
                path_image = os.path.join(self.root_dir, self.set_name, 'Annotations', image_id + '.xml')
                self.image_infos[image_id] = load_xml_meta(path_image)
            if not os.path.exists(os.path.join('.', 'cache')):
                os.makedirs(os.path.join('.', 'cache'))
            cache_ = open(path_, 'wb')
            pickle.dump(self.image_infos, cache_)
            cache_.close()
            
        print('loaded...')
        

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
        path_ = os.path.join(self.root_dir, self.set_name, 'JPEGImages', self.image_ids[image_index] + '.jpg')
        print(path_)
        img = cv2.imread(path_)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img

    def load_annotations(self, image_index):
        path_ = os.path.join(self.root_dir, self.set_name, 'Annotations', self.image_ids[image_index] + '.xml')
        
        return load_xml_annotaitons(path_, self.voc_labels)

    def voc_label_to_label(self, coco_label):
        return self.voc_labels_inverse[coco_label]

    def label_to_voc_label(self, label):
        return self.voc_labels[label]

    def image_aspect_ratio(self, image_index):
        image_info = self.image_infos[self.image_ids[image_index]]
        
        return float(image_info['width']) / float(image_info['height'])
    
    
if __name__ == '__main__':
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from dataset.transform import Normalizer, UniformResizer, ToTensor
    
    voc = VocDataset('/workspace/nas-data/dataset/voc/VOCdevkit', set_name='VOC2007', 
                       transform=transforms.Compose([Normalizer(), UniformResizer(), ToTensor()]))
    dataloader_train = DataLoader(voc)
    print(voc.voc_labels)
    print(voc.voc_labels_inverse)
    print(voc.image_ids[0])
    annot = load_xml_annotaitons(os.path.join(voc.root_dir, voc.set_name, 'Annotations', voc.image_ids[0] + '.xml'), voc.voc_labels)
    print(annot)
    for iter_num, data in enumerate(dataloader_train):
        print(data['image'].shape)
        if iter_num > 10:
            break
    