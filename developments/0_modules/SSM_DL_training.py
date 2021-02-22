#!/usr/bin/env python
# coding: utf-8

# In[1]:

from SSM_CNN import ConvNet
from SSM_utils import printProgress, Generate_folder, Batch_idxs

import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import time, datetime

import torch
from torch import nn
from torchsummary import summary
from torch import optim

import copy

class CNN_Training:
    def __init__(self, Xs, Ys, n_ch, n_cls, lr, n_batch_per_cls, n_epoch, n_patience, model_name, GPU_idx):
        self.Xs, self.Ys, self.n_ch, self.n_cls = Xs, Ys, n_ch, n_cls
        self.lr, self.n_batch, self.n_epoch, self.n_patience = lr, n_batch_per_cls, n_epoch, n_patience
        self.model_name = model_name
        self.GPU_idx = GPU_idx
    
    def Split2TV(self, data, label, rate_t_v = 0.9):
        data_num = len(data)
        train_idx = np.random.choice(data_num, int(rate_t_v*data_num), replace = False)
        valid_idx = np.setdiff1d(np.arange(data_num), train_idx)
        return data[train_idx], label[train_idx], data[valid_idx], label[valid_idx]
    
    def Get_device(self, GPU_idx = 3):
#         self.device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() else "cpu")
        print('')
        if torch.cuda.is_available() and type(GPU_idx) == int:
            self.device = torch.device("cuda:{}".format(GPU_idx))
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device), '\n')
        else:
            self.device = torch.device('cpu')
            print("Device: CPU\n")
            
    def Define_model_opt(self, n_ch, n_cls, lr = 0.00001, summary_show = True):
        self.model = ConvNet(n_ch, n_cls)
        # model = model.cuda()
        self.model = self.model.to(self.device)
        # if device == 'cuda':
        #     net = torch.nn.DataParallel(net)
        #     cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
#                                                                  mode='min',
#                                                                  factor=0.1,
#                                                                  patience=30)
        if summary_show == True:
            summary(self.model, (n_ch, 40, 40), device = self.device)
#         summary(self.model, (n_ch, 40, 40), device = 'cpu')
         
    def RandomMinibatch(self, data, label, n_batch = 100):
        if n_batch <= len(data):
            batch_idx = np.random.choice(len(data), n_batch, replace = False)
        else:
            batch_idx = np.random.choice(len(data), len(data), replace = False)
        return data[batch_idx], label[batch_idx]
    
    def Shuffle(self, x1, x2):
        """
        random shuffle of two paired data -> x, y = shuffle(x, y)
        but, available of one data -> x = shuffle(x, None)
        """
        idx = np.arange(len(x1))
        np.random.shuffle(idx)
        if type(x1) == type(x2):
            return x1[idx], x2[idx] 
        else:
            return x1[idx]
        
    def Torch_Minibatch_Load(self, Xs, Ys, batch_size = 100, shuffle = False):
        x, y = [], []
        for X, Y in zip(Xs, Ys):
            x_i, y_i = self.RandomMinibatch(X, Y, batch_size)
            x.append(x_i), y.append(y_i)
        x, y = np.concatenate(x), np.concatenate(y)
        if shuffle != False:
            x, y = self.Shuffle(x, y)
        x, y = torch.tensor(x, device = self.device).float(), torch.tensor(y, device = self.device).long()
#         x, y = torch.tensor(x, device = 'cpu').float(), torch.tensor(y, device = 'cpu').long()
        return x, y

    def Validation(self, X, Y, batch_size = 32):
        batch_idxs = Batch_idxs(X, batch_size)
        output = []
        for batch in batch_idxs:
            x = torch.tensor(X[batch, :, :, :], device = self.device).float() 
#             x = X[batch, :, :, :] 
            o = self.model(x)
            output.append(o)
        output = torch.cat(output)
        loss = self.criterion(output, Y)

        _, pred = torch.max(output, 1)
        return loss, pred   
    
    def Training_Process(self, n_epoch, n_batch, n_patience, verbose, model_name, save_path = './model/'):
    
        self.loss_hist, self.accr_hist = [], []
        self.val_loss_hist, self.val_accr_hist = [], []
        Generate_folder(save_path)
        
        iter_i = 0
        self.epoch_i = 0
        patience_i = 0
        
        while True:
            iter_i += 1

            train_x, train_y = self.Torch_Minibatch_Load(self.train_Xs, self.train_Ys, n_batch, shuffle = True)
            
            self.model.train()
            
            output = self.model(train_x)
            loss = self.criterion(output, train_y)
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
#             if iter_i % 10 == 0: 
            if iter_i % (np.max(self.n_data) // n_batch + 1) == 0:   
                self.epoch_i += 1
            
                self.model.eval()
                with torch.no_grad():
                    
                    train_loss, train_pred = self.Validation(self.train_X, self.train_Y, batch_size = 16)

                    self.loss_hist.append(train_loss.tolist())
                    self.accr_hist.append((torch.sum(train_pred == self.train_Y.data).tolist() / len(self.train_Y)))
                    
                    valid_loss, valid_pred = self.Validation(self.valid_X, self.valid_Y, batch_size = 16)
                    
                    self.val_loss_hist.append(valid_loss.tolist())
                    self.val_accr_hist.append((torch.sum(valid_pred == self.valid_Y.data).tolist() / len(self.valid_Y)))
                    
#                     self.scheduler.step(valid_loss.tolist())
                    
                if verbose == 1:
                    train_prt = 'train_loss: {:.5f}, train_accr: {:.3f}'.format(self.loss_hist[-1], self.accr_hist[-1])
                    val_prt = 'val_loss: {:.5f}, val_accr: {:.3f}'.format(self.val_loss_hist[-1], self.val_accr_hist[-1])
                    print("{:03d} | {} | {}".format(self.epoch_i, train_prt, val_prt))
                if verbose == 2:
                    printProgress(self.epoch_i, n_epoch, prefix = 'Epoch', suffix = str(n_epoch), barLength = 70)
                        
                if self.val_accr_hist[-1] == np.max(self.val_accr_hist):
                    patience_i = 0
                    
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    
#                     now = datetime.datetime.now()
#                     nowDatetime = now.strftime('%y%m%d%H%M')
                    params = '{}_{}'.format(self.optimizer.param_groups[0]['lr'], self.n_batch)
                    tr_spec = 't_accr_{:.4f}_t_loss_{:.6f}'.format(self.accr_hist[-1], self.loss_hist[-1])
                    vl_spec = 'v_accr_{:.4f}_v_loss_{:.6f}'.format(self.val_accr_hist[-1], self.val_loss_hist[-1])
#                     model_full_name = '{}_{}_{}_{:03d}_{}_{}.pt'.format(model_name, params, nowDatetime, self.epoch_i, tr_spec, vl_spec)
                    
                else:
                    patience_i += 1
                        
            if self.epoch_i == n_epoch or patience_i == n_patience:
                self.model.load_state_dict(best_model_wts)
#                 torch.save(self.model.state_dict(), save_path + model_full_name)
                torch.save({
                    'epoch': self.epoch_i,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'batch': self.n_batch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'tr_spec': tr_spec,
                    'vl_spec': vl_spec}, 
                    save_path + '{}_{}_{}.pt'.format(model_name, tr_spec, vl_spec))
                
                
                break
                    
    def PlotHIST(self, model_name, save_path = './results/npy/'):
        
        fig = plt.figure(figsize = (20, 8))
        
        # x_axis = range(1, 10*len(accr_hist)+1, 10)
#         x_axis = np.arange(10, 10*len(self.accr_hist)+1, 10)
        x_axis = np.arange(1, self.epoch_i + 1)

        plt.subplot(1, 2, 1)
        plt.plot(x_axis, self.accr_hist, 'b-', label = 'Training Accuracy')
        plt.plot(x_axis, self.val_accr_hist, 'r-', label = 'Validation Accuracy')
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('Accuracy', fontsize = 15)
        plt.legend(fontsize = 10)
        plt.grid('on')
        plt.subplot(1, 2, 2)
        plt.plot(x_axis, self.loss_hist, 'b-', label = 'Training Loss')
        plt.plot(x_axis, self.val_loss_hist, 'r-', label = 'Validation Loss')
        plt.xlabel('Epoch', fontsize = 15)
        plt.ylabel('Loss', fontsize = 15)
        # plt.yticks(np.arange(0, 0.25, step=0.025))
        plt.legend(fontsize = 12)
        plt.grid('on')
        plt.show()
        
        Generate_folder(save_path)
        
        np.save(save_path + '{}_accr_hist'.format(model_name), self.accr_hist)
        np.save(save_path + '{}_val_accr_hist'.format(model_name), self.val_accr_hist)
        np.save(save_path + '{}_loss_hist'.format(model_name), self.loss_hist)
        np.save(save_path + '{}_val_loss_hist'.format(model_name), self.val_loss_hist)
        
    def ModelSelection(self, model_dir, model_name):
        model_list = np.array([i for i in os.listdir(model_dir) if model_name + '_' in i])
        n_model = len(model_list)
        v_accr, t_accr, v_loss = np.zeros([n_model]), np.zeros([n_model]), np.zeros([n_model])
        for i, file in zip(range(n_model), model_list):
            v_accr[i] = file.split('v_accr')[-1].split('_')[1]
            t_accr[i] = file.split('t_accr')[-1].split('_')[1]
            v_loss[i] = file.split('v_loss')[-1].split('_')[1][:-3]
        t_bigger_v = np.where(t_accr >= v_accr)[0]
        if len(t_bigger_v) != 0:
            v_max = np.where(v_accr == np.max(v_accr[t_bigger_v]))[0]
            t_max = np.where(t_accr == np.max(t_accr[v_max]))[0]
            v_min = np.where(v_loss == np.min(v_loss[t_max]))[0]
        elif len(t_bigger_v) == 0:
            v_max = np.where(v_accr == np.max(v_accr))[0]
            t_max = np.where(t_accr == np.max(t_accr[v_max]))[0]
            v_min = np.where(v_loss == np.min(v_loss[t_max]))[0] 
        if len(v_min) == 1:
            best_idx = v_min
        elif len(v_min) != 1:
            best_idx = v_min[0:1]
        delete_idx = np.setdiff1d(np.arange(n_model), best_idx)
        for i in delete_idx:
            os.remove(model_dir + model_list[i])
        print('Best Model:', model_list[int(best_idx)])
          
    def Run(self, model_spec = 1, verbose = 0, model_dir = './model/'):
        self.train_Xs, self.train_Ys = [], []
        self.valid_X, self.valid_Y = [], []
        self.n_data = []
        for i, X, Y in zip(range(self.n_cls), self.Xs, self.Ys):
            print(i, X.shape, Y.shape)
            train_x, train_y, valid_x, valid_y = self.Split2TV(X, Y, rate_t_v = 0.9)
            print(i, 'train:', train_x.shape, train_y.shape, 'valid:', valid_x.shape, valid_y.shape)
            self.train_Xs.append(train_x), self.train_Ys.append(train_y)
            self.valid_X.append(valid_x), self.valid_Y.append(valid_y)
            self.n_data.append(len(train_x))    
        
        self.Get_device(self.GPU_idx)
        self.Define_model_opt(self.n_ch, self.n_cls, self.lr, summary_show = model_spec)
        
        # for validation
        self.train_X, self.train_Y = np.concatenate(self.train_Xs), np.concatenate(self.train_Ys)
#         self.train_X = torch.tensor(self.train_X, device = self.device).float() 
        self.train_Y = torch.tensor(self.train_Y, device = self.device).long()
        
        self.valid_X, self.valid_Y = np.concatenate(self.valid_X), np.concatenate(self.valid_Y)
#         self.valid_Xs = torch.tensor(self.valid_Xs, device = self.device).float() 
        self.valid_Y = torch.tensor(self.valid_Y, device = self.device).long()
        
        self.Training_Process(self.n_epoch, self.n_batch, self.n_patience, verbose, self.model_name, save_path = model_dir)
        self.PlotHIST(self.model_name)    
        self.ModelSelection(model_dir, self.model_name)

