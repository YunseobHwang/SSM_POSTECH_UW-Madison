#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os
import pickle
import gzip

def printProgress(iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
#     percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r{} |{} | {} / {}'.format(prefix, bar, iteration, suffix)),
#     sys.stdout.write('\r{} |{} | {}{} {}'.format(prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def Generate_folder(newpath):
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
def DataConfig(data_dir):
    data_config = {}
    data_config['GOOD'], data_config['BAD'] = [], []
    
    # DIVIDE GOOD AND BAD
    for file in os.listdir(data_dir):
        if 'GOOD' in file:
            data_config['GOOD'].append(file)
        elif 'BAD' in file:
            data_config['BAD'].append(file)
            
    # LOAD NPY FROM EACH FILE
    for cls, files in data_config.items():
        for i, file in enumerate(files):
            data = np.load(os.path.join(data_dir, file))
            data_config[cls][int(i)] = {'0_index': int(i+1), '1_file': file, '2_data': data}
    
    # TRAINING DATA / TESTING DATA SPLIT
    
    for cls, files in data_config.items():
        for i, file in enumerate(files):
            data = data_config[cls][int(i)]['2_data']
            data_num = len(data)
            train_idx = np.random.choice(data_num, int(0.8*data_num), replace = False)
            train_data = data[train_idx]
            test_data = data[np.setdiff1d(np.arange(data_num), train_idx)]
            data_config[cls][int(i)]['3_train_data'], data_config[cls][int(i)]['4_test_data'] = train_data, test_data
    with gzip.open(data_dir + '/SemblexData_config.pickle', 'wb') as f:
        pickle.dump(data_config, f)

def concat(data):
    return np.concatenate(data)

def add_ch(img):
    """
    (sample #, height, width,) -> (sample #, height, width, channel)
    """
    return np.expand_dims(img, axis = -1)

def Reshape4torch(img):
    """
    (sample #, height, width, channel) -> (sample #, channel, height, width)
    """
    img = np.transpose(img, (0, 3, 1, 2))
    return img
    
def GenerateLabel(data, cls):
    label = cls*np.ones([data.shape[0]])
    return label

def GBdataLoad(data_dir, ch = [0, 1], data_type = '3_train_data'):
    
    with gzip.open(data_dir + '/SemblexData_config.pickle', 'rb') as f:
        data = pickle.load(f)
       
    G_X, B_X = [], []
    for cls, data_bunch in data.items():
        for data_i in data_bunch:
            if cls == 'GOOD':
                G_X.append(data_i[data_type][:, :, :, ch])
            elif cls == 'BAD':
                B_X.append(data_i[data_type][:, :, :, ch])
     
    G_X, B_X = concat(G_X), concat(B_X)
    
    if len(G_X.shape) != 4:
        G_X, B_X = add_ch(G_X), add_ch(B_X)
    G_X, B_X = Reshape4torch(G_X), Reshape4torch(B_X)
    GB_Xs, GB_Ys = [G_X, B_X], []
    for i, GB_X in zip(range(len(GB_Xs)), GB_Xs):
        GB_Ys.append(GenerateLabel(GB_X, i))
        
    if 'test' in data_type:
        return concat(GB_Xs), concat(GB_Ys)
    else:
        return GB_Xs, GB_Ys
        
def B6dataLoad(data_dir, ch = [0, 1], data_type = '3_train_data'):
    
    with gzip.open(data_dir + '/SemblexData_config.pickle', 'rb') as f:
        data = pickle.load(f)
        
    BAD_cls = {0: 'OIL', 1: 'PUNCH', 2: 'SCRAPPED', 3: 'DIE_CHIP', 4: 'DIE_INTERNAL', 5: 'PIN'}

    B_X1, B_X2, B_X3, B_X4, B_X5, B_X6 = [], [], [], [], [], []
    for cls, data_bunch in data.items():
        for data_i in data_bunch:
            if cls == 'BAD':
                if BAD_cls[0] in data_i['1_file']: B_X1.append(data_i[data_type][:, :, :, ch])
                if BAD_cls[1] in data_i['1_file']: B_X2.append(data_i[data_type][:, :, :, ch])
                if BAD_cls[2] in data_i['1_file']: B_X3.append(data_i[data_type][:, :, :, ch])
                if BAD_cls[3] in data_i['1_file']: B_X4.append(data_i[data_type][:, :, :, ch])
                if BAD_cls[4] in data_i['1_file']: B_X5.append(data_i[data_type][:, :, :, ch])
                if BAD_cls[5] in data_i['1_file']: B_X6.append(data_i[data_type][:, :, :, ch])

    B_X1, B_X2, B_X3, B_X4, B_X5, B_X6 = concat(B_X1), concat(B_X2), concat(B_X3), concat(B_X4), concat(B_X5), concat(B_X6)
    
    if len(B_X1.shape) != 4:
        B_X1, B_X2, B_X3, B_X4, B_X5, B_X6 = add_ch(B_X1), add_ch(B_X2), add_ch(B_X3), add_ch(B_X4), add_ch(B_X5), add_ch(B_X6)
    
    B_X1, B_X2, B_X3, B_X4, B_X5, B_X6 = (Reshape4torch(B_X1), Reshape4torch(B_X2), Reshape4torch(B_X3), 
                                          Reshape4torch(B_X4), Reshape4torch(B_X5), Reshape4torch(B_X6))
    
    B6_Xs = [B_X1, B_X2, B_X3, B_X4, B_X5, B_X6]
    B6_Ys = []
    for i, B6_X in zip(range(len(B6_Xs)), B6_Xs):
        B6_Ys.append(GenerateLabel(B6_X, i))
    if 'test' in data_type:
        return concat(B6_Xs), concat(B6_Ys)
    else:
        return B6_Xs, B6_Ys
    
def Batch_idxs(data, batch_size = 250):
    """generate the serial batch of data on index-level.
       Usually, the data is too large to be evaluated at once.
    
    Args:
      data: A list or array of target dataset e.g. data_x we use
      batchsize: A integer
      
    Returns:
      batch_idxs: A list, 
    """
    total_size = len(data)
    batch_idxs = []
    start = 0
    while True:
        if total_size >= start + batch_size:
            batch_idxs.append([start + i for i in range(batch_size)])
        elif total_size < start + batch_size:
            batch_idxs.append([start + i for i in range(total_size - start)])
        start += batch_size
        if total_size <= start:
            break
    return batch_idxs