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
    def __init__(self, test_X, test_Y, n_ch, n_cls, class_name, model_dir, model_name, batch_size, GPU_idx):
        self.test_X, self.test_Y = test_X, test_Y
        self.n_ch, self.n_cls, self.class_name = n_ch, n_cls, class_name
        self.model_dir, self.model_name = model_dir, model_name
        self.batch_size, self.GPU_idx = batch_size, GPU_idx
        
        self.model_list = self.Model_find(self.model_dir, self.model_name)
        self.Get_device(self.GPU_idx)
            
    def Get_device(self, GPU_idx = 3):
#         self.device = torch.device("cuda:{}".format(GPU_idx) if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and type(GPU_idx) == int:
            self.device = torch.device("cuda:{}".format(GPU_idx))
            current_device = torch.cuda.current_device()
#             print("Device:", torch.cuda.get_device_name(current_device), '\n')
        else:
            self.device = torch.device('cpu')
            print("Device: CPU\n")
            
    def Model_find(self, model_dir, model_name):
        model_list = [i for i in sorted(os.listdir(model_dir)) if model_name + '_' in i]
        
        return model_list   
    
    def Model_import(self, n_ch, n_cls, model_dir, model_file):
        self.model = ConvNet(n_ch, n_cls)
        self.model = self.model.to(self.device)
#         self.model.load_state_dict(torch.load(model_dir + model_file))
#         self.model.load_state_dict(torch.load(model_dir + model_file, map_location=lambda storage, loc: storage.cuda(self.GPU_idx)))
    
        self.ckpt = torch.load(model_dir + model_file, map_location=lambda storage, loc:storage.cuda(self.GPU_idx))
        self.model.load_state_dict(self.ckpt['model_state_dict'])
        self.epoch = self.ckpt['epoch']
        self.lr = self.ckpt['lr']
        
        self.model.eval()

    def Model_pred(self, model, test_X, test_Y):
        test_X, test_Y = torch.tensor(test_X, device=self.device).float(), torch.tensor(test_Y, device=self.device).long()
        output = model(test_X)

        _, pred = torch.max(output, 1)
        
#         return torch.tensor(output, device = 'cpu').numpy(), pred.tolist()
        return output.cpu().detach().numpy(), pred.tolist()
        
    
    def Model_accr(self, test_Y, pred):
        num_correct = np.sum(test_Y == pred)
        accr = round(100 * num_correct / len(test_Y), 2)
        print("Accuracy: {:.2f} % ({} / {})\n".format(accr, num_correct, len(test_Y)))
        return accr
    
    def ConfusionMatrix(self, save_path = './results/1_confusion_mat/', save_name = None, x_angle = 0):
    
        def plotCM(cm, value_size, mode):

            plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Blues)
            thresh = cm.max()/2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                if mode == 'percent':
                    value = np.round(cm[i, j]/(np.sum(cm, 1)[i]), 3)
                elif mode == 'num':
                    value = cm[i, j]
                plt.text(j, i, value,
                         fontsize = value_size,
                         horizontalalignment = 'center',
                         color = 'white' if cm[i, j] > thresh else 'black')
            plt.ylabel('Actual', fontsize = 20)
            plt.xlabel('Predicted', fontsize = 20)
            plt.xticks([i for i in range(len(self.class_name))], self.class_name, rotation= x_angle, fontsize = 18)
            plt.yticks([i for i in range(len(self.class_name))], self.class_name, rotation=0, fontsize = 18)

        self.cm = confusion_matrix(self.test_Y, self.pred)
        
        if save_path != None:
            Generate_folder(save_path)
        
        if save_name == None:
            save_name = self.model_name
        
        modes = ['percent', 'num']
        for i, m in zip(range(len(modes)), modes):
            fig = plt.figure(figsize = (13, 10))
            plotCM(self.cm, value_size = 22, mode = m)   
            plt.show()
            if save_path != None:
                fig.savefig(save_path + '{}_CM_{}.png'.format(save_name, m), bbox_inches='tight')
            plt.close(fig)
    
    def compute_cls_metrics(self):
        # 'Accuracy', 'Precision', 'Recall', 'F1 score'
        
        self.cm = confusion_matrix(self.test_Y, self.pred)
        
        if self.cm.shape[0] == 2:
            accr = round((self.cm[0, 0] + self.cm[1, 1]) / np.sum(self.cm), 4)
            precision = round(self.cm[1, 1] / (self.cm[0, 1] + self.cm[1, 1]), 4)
            recall = round(self.cm[1, 1] / (self.cm[1, 0] + self.cm[1, 1]), 4)
            f1_score = round(2*precision*recall/(precision + recall), 4)
            self.cls_metrics = [accr, precision, recall, f1_score]
            return self.cls_metrics
        
        else:
            self.cls_metrics = []
            for i in range(self.cm.shape[0]):
                accr = round(np.trace(self.cm) / np.sum(self.cm), 4)
                precision = round(self.cm[i, i] / np.sum(self.cm[:, i]), 4)
                recall = round(self.cm[i, i] / np.sum(self.cm[i, :]), 4)
                f1_score = round(2*precision*recall/(precision + recall), 4)
                self.cls_metrics.append([accr, precision, recall, f1_score])     
            return self.cls_metrics 
        
    def Inference(self, model_file, batch_size):
        
        self.Model_import(self.n_ch, self.n_cls, self.model_dir, model_file)
        self.batch_idxs = Batch_idxs(self.test_X, batch_size = batch_size)
        self.score, self.pred = [], []
        for batch in self.batch_idxs:   
            score, pred = self.Model_pred(self.model, self.test_X[batch, :, :, :], self.test_Y[batch])
            self.score.append(score), self.pred.append(pred)
        self.score, self.pred = np.concatenate(self.score), np.concatenate(self.pred)
    
    def Evaluation(self):
        for model_file in self.model_list:
            self.Inference(model_file, batch_size = self.batch_size)
            print('model:', model_file)
            print('epoch: {:d}, lr (starting at 0.01): {}'.format(self.epoch, self.lr))
#             self.accr = self.Model_accr(self.test_Y, self.pred)
            self.compute_cls_metrics()
                  