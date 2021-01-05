#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-17 22:19:38
LastEditTime: 2021-01-05 10:23:50
@Description: file content
'''
from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from data.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np

class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net
        
        self.model = net(
                num_channels=self.cfg['data']['n_colors'], 
                base_filter=64,  
                scale_factor=self.cfg['data']['upsacle'], 
                args = self.cfg
        )

    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0]) 
            
            self.model_path = os.path.join(self.cfg['checkpoint'], self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids)
            self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage)['net'])

    def test(self):
        self.model.eval()
        avg_time= []
        for batch in self.data_loader:
            with torch.no_grad():
                input, label, name = Variable(batch[0]), Variable(batch[1]), batch[2]
            noise = torch.FloatTensor(input.size()).normal_(mean=0, std=int(self.cfg['data']['noise'])/255.).float() 
            if self.cuda:
                input, label = input.cuda(), label.cuda()
                noise = noise.cuda()
            input = input + noise
            t0 = time.time()
            prediction = self.model(input)   
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            np.save(name[0][0:-4]+'_input.npy',input.cpu().data)
            np.save(name[0][0:-4]+'_target.npy',target.cpu().data)
            np.save(name[0][0:-4]+'_prediction.npy',prediction.cpu().data)
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))

   
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':            
            self.dataset = get_test_data(self.cfg, self.cfg['test']['test_dataset'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.test()
        elif self.cfg['test']['type'] == 'eval':            
            raise ValueError('Mode error!')
        else:
            raise ValueError('Mode error!')