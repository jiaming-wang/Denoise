#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-13 23:04:48
LastEditTime: 2020-12-31 16:36:45
@Description: file content
'''
import os, importlib, torch, shutil
from solver.basesolver import BaseSolver
from utils.utils import maek_optimizer, make_loss, calculate_psnr, calculate_ssim, save_config, save_net_config, save_net_py
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import numpy as np
from importlib import import_module
from torch.autograd import Variable
from data.data import DatasetFromHdf5
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils.config import save_yml

class Solver(BaseSolver):
    def __init__(self, cfg):
        super(Solver, self).__init__(cfg)
        self.init_epoch = self.cfg['schedule']
        
        net_name = self.cfg['algorithm'].lower()
        lib = importlib.import_module('model.' + net_name)
        net = lib.Net

        self.model = net(
            num_channels=self.cfg['data']['n_colors'], 
            base_filter=64,  
            scale_factor=self.cfg['data']['upsacle'], 
            args = self.cfg
        )
        self.model = self.model.double() 
        self.optimizer = maek_optimizer(self.cfg['schedule']['optimizer'], cfg, self.model.parameters())
        self.loss = make_loss(self.cfg['schedule']['loss'])

        self.log_name = self.cfg['algorithm'] + '_' + str(self.timestamp)
        # save log
        self.writer = SummaryWriter('log/' + str(self.log_name))
        save_net_config(self.log_name, self.model)
        save_net_py(self.log_name, net_name)
        save_yml(cfg, os.path.join('log/' + str(self.log_name), 'config.yml'))
        save_config(self.log_name, 'Train dataset has {} images and {} batches.'.format(len(self.train_dataset), len(self.train_loader)))
        save_config(self.log_name, 'Val dataset has {} images and {} batches.'.format(len(self.val_dataset), len(self.val_loader)))
        save_config(self.log_name, 'Model parameters: '+ str(sum(param.numel() for param in self.model.parameters())))

    def train(self): 
        with tqdm(total=len(self.train_loader), miniters=1,
                desc='Initial Training Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t:

            epoch_loss = 0
            for iteration, batch in enumerate(self.train_loader, 1):
                lr, hr = Variable(batch[0]), Variable(batch[1])
            #for data in self.train_loader:
                #lr, hr = data[0], data[1]
                if self.cuda:
                    lr, hr = lr.cuda(self.gpu_ids[0]), hr.cuda(self.gpu_ids[0])
                
                self.optimizer.zero_grad()               
                self.model.train()

                sr = self.model(lr)
                loss = self.loss(sr, hr) / (self.cfg['data']['batch_size'] * 2)

                epoch_loss += loss.data
                t.set_postfix_str("Batch loss {:.4f}".format(loss.item()))
                t.update()

                loss.backward()
                # print("grad before clip:"+str(self.model.output_conv.conv.weight.grad))
                if self.cfg['schedule']['gclip'] > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg['schedule']['gclip']
                    )
                self.optimizer.step()
                
            self.records['Loss'].append(epoch_loss / len(self.train_loader))
            save_config(self.log_name, 'Initial Training Epoch {}: Loss={:.4f}'.format(self.epoch, self.records['Loss'][-1]))
            self.writer.add_scalar('Loss_epoch', self.records['Loss'][-1], self.epoch)

    def eval(self):
        with tqdm(total=len(self.val_loader), miniters=1,
                desc='Val Epoch: [{}/{}]'.format(self.epoch, self.nEpochs)) as t1:
            psnr_list, ssim_list = [], []
            for iteration, batch in enumerate(self.val_loader, 1):
                lr, hr = Variable(batch[0]), Variable(batch[1])

                if self.cuda:
                    lr, hr = lr.cuda(), hr.cuda()
                self.model.eval()
                with torch.no_grad():
                    sr = self.model(lr)
                    loss = self.loss(sr, hr)

                batch_psnr, batch_ssim = [], []
                for c in range(sr.shape[0]):
                    #predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    #ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    if not self.cfg['data']['normalize']:
                        predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                        ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0))) * 255
                    else:          
                        predict_sr = (sr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                        ground_truth = (hr[c, ...].cpu().numpy().transpose((1, 2, 0)) + 1) * 127.5
                    psnr = calculate_psnr(predict_sr, ground_truth, 255)
                    ssim = calculate_ssim(predict_sr, ground_truth, 255)
                    batch_psnr.append(psnr)
                    batch_ssim.append(ssim)
                avg_psnr = np.array(batch_psnr).mean()
                avg_ssim = np.array(batch_ssim).mean()
                psnr_list.extend(batch_psnr)
                ssim_list.extend(batch_ssim)
                t1.set_postfix_str('Batch loss: {:.4f}'.format(loss.item()))
                t1.update()
            self.records['Epoch'].append(self.epoch)

    def check_gpu(self):
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
            self.loss = self.loss.cuda(self.gpu_ids[0])

            self.model = self.model.cuda(self.gpu_ids[0])
            self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_ids) 

    def check_pretrained(self):
        checkpoint = os.path.join(self.cfg['pretrain']['pre_folder'], self.cfg['pretrain']['pre_sr'])
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage)['net'])
            self.epoch = torch.load(checkpoint, map_location=lambda storage, loc: storage)['epoch']
            if self.epoch > self.nEpochs:
                raise Exception("Pretrain epoch must less than the max epoch!")
        else:
            raise Exception("Pretrain path error!")

    def save_checkpoint(self):
        super(Solver, self).save_checkpoint()
        self.ckp['net'] = self.model.state_dict()
        self.ckp['optimizer'] = self.optimizer.state_dict()
        if not os.path.exists(self.cfg['checkpoint'] + '/' + str(self.log_name)):
            os.mkdir(self.cfg['checkpoint'] + '/' + str(self.log_name))
        torch.save(self.ckp, os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'))

        if self.cfg['save_best']:
            if self.records['Loss'] != [] and self.records['Loss'][-1] == np.array(self.records['Loss']).min():
                shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
                            os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best.pth'))
            # if self.records['PSNR'] != [] and self.records['PSNR'][-1] == np.array(self.records['PSNR']).max():
            #     shutil.copy(os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'latest.pth'),
            #                 os.path.join(self.cfg['checkpoint'] + '/' + str(self.log_name), 'best.pth'))

    def run(self):
        self.check_gpu()
        if self.cfg['pretrain']['pretrained']:
            self.check_pretrained()
        try:
            while self.epoch <= self.nEpochs:
                self.train()
                self.eval()
                self.save_checkpoint()
                self.epoch += 1
        except KeyboardInterrupt:
            self.save_checkpoint()
        save_config(self.log_name, 'Training done.')