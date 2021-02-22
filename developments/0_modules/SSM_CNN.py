#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self, n_ch, n_cls):
        super().__init__()

        self.conv1_1 = nn.Conv2d(n_ch, 64, 3, 1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.maxp1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.maxp2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.maxp3 = nn.MaxPool2d(2, 2)
        
        self.conv4_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.maxp4 = nn.MaxPool2d(2, 2)
        
        self.dense1 = nn.Linear(2*2*64, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, n_cls)  
        
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_bn(self.conv1_2(x)))
        x = self.maxp1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_bn(self.conv2_2(x)))
        x = self.maxp2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_bn(self.conv3_2(x)))
        x = self.maxp3(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_bn(self.conv4_2(x)))
        x = self.maxp4(x)
        # flatten
        x = x.view(-1, 2*2*64)
        feature = F.relu(self.dense1(x))
        x = self.dropout1(feature)
        x = self.dense2(x)
        return x

# CNN model for part classification previsouly designed by Kangsan
class PCNet(nn.Module):
    def __init__(self, n_ch, n_cls):
        super().__init__()

        self.conv1_1 = nn.Conv2d(n_ch, 64, 3, 1, padding=1)
        self.conv1_1bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv1_2bn = nn.BatchNorm2d(64)
        
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv2_1bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv2_2bn = nn.BatchNorm2d(64)
        
        self.conv3_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3_1bn = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv3_2bn = nn.BatchNorm2d(64)
        
        self.conv4_1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        
        self.dense1 = nn.Linear(5*5*64, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(128, n_cls)  
        
    def forward(self, x):
        x = self.conv1_1bn(F.relu(self.conv1_1(x)))
        x = F.relu(self.conv1_2(x))
        x = self.conv1_2bn(self.maxp1(x))
        x = self.conv2_1bn(F.relu(self.conv2_1(x)))
        x = F.relu(self.conv2_2(x))
        x = self.conv2_2bn(self.maxp2(x))
        x = self.conv3_1bn(F.relu(self.conv3_1(x)))
        x = F.relu(self.conv3_2(x))
        x = self.conv3_2bn(self.maxp3(x))
        x = self.conv4_bn(F.relu(self.conv4_1(x)))
        x = F.relu(self.conv4_2(x))
        # flatten
        x = x.view(-1, 5*5*64)
        feature = F.relu(self.dense1(x))
        x = self.dropout1(feature)
        x = self.dense2(x)
        return x