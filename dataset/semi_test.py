from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ColorJitter, RandomGrayscale

import torch
from torchvision.transforms import ColorJitter, RandomGrayscale, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomRotation


def augmentation_transform(img, mask):
    # 随机裁剪
    transform = RandomCrop(size=(256, 256))
    img = transform(img)
    mask = transform(mask)

    # 随机水平翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # 随机垂直翻转
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # 随机旋转
    angle = random.randint(-15, 15)
    img = img.rotate(angle)
    mask = mask.rotate(angle)

    # 颜色扰动
    img = ColorJitter(0.5, 0.5, 0.5, 0.25)(img)

    # 随机灰度化
    if random.random() < 0.2:
        img = RandomGrayscale(p=1.0)(img)
    if img.size != mask.size:
        img = img.resize(mask.size, Image.BILINEAR)

    return img, mask



class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'test':  # Added support for the test dataset
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id
        if self.mode == 'test':
            # For the test dataset, you can perform any necessary processing here
            # Example: img, mask = your_test_data_processing_function(img, mask)
            img, mask = normalize(img, mask)
            return img, mask, id  # Return the test data

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img, mask = augmentation_transform(img, mask)  # 使用数据增强
            img, mask = normalize(img, mask)
            return img, mask


        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))


        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
