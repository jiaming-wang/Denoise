#!/usr/bin/env python
# coding=utf-8
'''
Author: wjm
Date: 2020-12-24 22:19:21
LastEditTime: 2021-01-05 10:43:48
Description: file content
'''
import numpy as np
import scipy.io as sio

input_data = np.load(r"1_prediction.npy")
input_data = np.squeeze(input_data).transpose(1,2,0)
sio.savemat('prediction.mat', {'prediction':input_data})
