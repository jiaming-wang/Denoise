#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-05 10:09:29
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
import scipy.io as sio 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',])


def load_img(filepath):
    img = sio.loadmat(filepath)
    #img = Image.open(filepath)
    #y, _, _ = img.split()
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(img_in, img_tar, img_bic, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))
    img_bic = img_bic.crop((ty,tx,ty + tp, tx + tp))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_bic, info_patch

def augment(img_in, img_tar, img_bic, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_bic = ImageOps.flip(img_bic)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_bic = ImageOps.mirror(img_bic)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_bic = img_bic.rotate(180)
            info_aug['trans'] = True
            
    return img_in, img_tar, img_bic, info_aug

class Data(data.Dataset):
    def __init__(self, image_dir, cfg, transform=None):
        super(Data, self).__init__()
        input_dir = image_dir + '/input'
        label_dir = image_dir + '/label'
        self.input_image_filenames = [join(input_dir, x) for x in listdir(input_dir)]
        self.label_image_filenames = [join(label_dir, x) for x in listdir(label_dir)]
        self.patch_size = cfg['data']['patch_size']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        label = load_img(self.label_image_filenames[index])['label']
        input = load_img(self.input_image_filenames[index])['input']

        _, file = os.path.split(self.label_image_filenames[index])

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
            
        return input, label, file

    def __len__(self):
        return len(self.input_image_filenames)

class Data_test(data.Dataset):
    def __init__(self, image_dir, cfg, transform=None):
        super(Data_test, self).__init__()
        
        input_dir = image_dir + '/input'
        label_dir = image_dir + '/label'
        self.input_image_filenames = [join(input_dir, x) for x in listdir(input_dir)]
        self.label_image_filenames = [join(label_dir, x) for x in listdir(label_dir)]
        self.patch_size = cfg['data']['patch_size']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        label = load_img(self.label_image_filenames[index])['label']
        input = load_img(self.input_image_filenames[index])['input']

        _, file = os.path.split(self.label_image_filenames[index])

        if self.transform:
            input = self.transform(input)
            label = self.transform(label)
            
        return input, label, file

    def __len__(self):
        return len(self.input_image_filenames)
