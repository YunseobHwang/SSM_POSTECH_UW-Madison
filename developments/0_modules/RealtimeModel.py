#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from SSM_CNN import ConvNet
from SSM_utils import Generate_folder, Batch_idxs

import os
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch

from sklearn.metrics import confusion_matrix
import itertools

class SSM_Model:
    def __init__(self, n_ch, n_cls, class_name, model_dir, model_name, batch_size, GPU_idx):
        self.n_ch, self.n_cls, self.class_name = n_ch, n_cls, class_name
        self.model_dir, self.model_name = model_dir, model_name
        self.batch_size, self.GPU_idx = batch_size, GPU_idx
        
        self.model_file = self.Model_find(self.model_dir, self.model_name)
        self.Get_device(self.GPU_idx)
        self.Model_import(self.n_ch, self.n_cls, self.model_dir, self.model_file)
            
    def Get_device(self, GPU_idx = 3):
#         self.device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() else "cpu")
        print('')
        if torch.cuda.is_available() and type(GPU_idx) == int:
            self.device = torch.device("cuda:{}".format(GPU_idx))
            current_device = torch.cuda.current_device()
            print("Device:", torch.cuda.get_device_name(current_device), '\n')
        else:
            self.device = torch.device('cpu')
#             print("Device:", torch.cuda.get_device_name(current_device), '\n')
            print("Device: CPU\n")
            
    def Model_find(self, model_dir, model_name):
        model_file = [i for i in sorted(os.listdir(model_dir)) if model_name + '_' in i][0]
        print('Model:', model_file, '\n')
        return model_file   
    
    def Model_import(self, n_ch, n_cls, model_dir, model_file):
        self.model = ConvNet(n_ch, n_cls)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(model_dir + model_file))
#         self.model.load_state_dict(torch.load(model_dir + model_file, map_location=lambda storage, loc: storage.cuda(self.GPU_idx)))
        self.model.eval()

    def Model_pred(self, model, test_X):
        test_X = torch.tensor(test_X, device=self.device).float()
        output = model(test_X)
        
        _, pred = torch.max(output, 1)
        output = torch.nn.functional.softmax(output, 1)
#         return torch.tensor(output, device = 'cpu').numpy(), pred.tolist()
        return output.cpu().detach().numpy(), pred.tolist()
        
    def Inference(self, test_X):
        
        self.score, self.pred = self.Model_pred(self.model, test_X)