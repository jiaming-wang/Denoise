#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2020-12-31 23:17:44
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .dataset import Data, Data_test
from torchvision import transforms
import torch, h5py, numpy
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, data_dir):
    data_dir = join(cfg['data_dir'], data_dir)
    cfg = cfg
    return Data(data_dir, cfg, transform=transform())
    
def get_test_data(cfg, data_dir):
    data_dir = join(cfg['test']['data_dir'], data_dir)
    return Data_test(data_dir, cfg, transform=transform())