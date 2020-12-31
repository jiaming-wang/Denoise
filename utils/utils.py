#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:12:52
LastEditTime: 2020-12-31 16:46:55
@Description: file content
'''
import os, math, torch, cv2, shutil
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from utils.vgg import VGG
import torch.nn.functional as F

def maek_optimizer(opt_type, cfg, params):
    if opt_type == "ADAM":
        optimizer = torch.optim.Adam(params, lr=cfg['schedule']['lr'], betas=(cfg['schedule']['beta1'], cfg['schedule']['beta2']), eps=cfg['schedule']['epsilon'])
    elif opt_type == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg['schedule']['lr'], momentum=cfg['schedule']['momentum'])
    elif opt_type == "RMSprop":
        optimizer = torch.optim.RMSprop(params, lr=cfg['schedule']['lr'], alpha=cfg['schedule']['alpha'])
    else:
        raise ValueError
    return optimizer

def make_loss(loss_type):
    # loss = {}
    if loss_type == "MSE":
        loss = nn.MSELoss(size_average=False)
    elif loss_type == "L1":
        loss = nn.L1Loss(size_average=False)
    elif loss_type == "VGG22":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "VGG54":
        loss = VGG(loss_type[3:], rgb_range=255)
    elif loss_type == "Cycle":
        loss = CycleLoss()
    else:
        raise ValueError
    return loss

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class CycleLoss(nn.Module):
    def __init__(self, loss_type = 'L1'):

        if loss_type == "MSE":
            self.loss = nn.MSELoss(reduction='sum')
        elif loss_type == "L1":
            self.loss = nn.L1Loss(reduction='sum')
        else:
            raise ValueError

    def forward(self, x_hr, x_lr, scale = 1/4):
        down_x = F.interpolate(x_hr, scale_factor=scale, mode='bicubic')
        return self.loss(down_x, x_lr)
        
def get_path(subdir):
    return os.path.join(subdir)

def save_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/records.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/records.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_config(time, log):
    open_type = 'a' if os.path.exists(get_path('./log/' + str(time) + '/net.txt'))else 'w'
    log_file = open(get_path('./log/' + str(time) + '/net.txt'), open_type)
    log_file.write(str(log) + '\n')

def save_net_py(time, py):
    py_path = os.path.join('./model', py+'.py')
    shutil.copyfile(py_path, os.path.join('./log/'+ str(time), py+'.py'))