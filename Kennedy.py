#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 17:59:29 2018

@author: Edvin
"""

#imports
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
#%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import numpy as np

#files
data_dir = "./data"
training_file = "senti_binary.train"
test_file = "senti_binary.test"
word_file = "wv_50d.txt"
batch_size = 10

#dataloader
class Family_Mart(Dataset):
    
    def __init__(self, data_dir, data_file, word_file):
        
        data = self._get_data(data_dir, data_file)
        word = self._get_word(data_dir, word_file)
        
        self.data = self._convert_data(data, word)
        
    def __getitem__(self, index):
        
        return self.data[Index]
    
    def __len__(self):
        
        return len(self.data)
    
    def _get_data(self, data_dir, file_name):
        
        data_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(data_path):
            print("Perkele vitto")
            return None
        
        data = []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                text = [w.lower().strip() for w in line.split()[:-1]]
                sentiment = int(line.split([-1].strip())
                example = (text, sentiment)
                data.append(example)  
                
        return data
    
    def _get_word(self, data_dir, word_file):
        
        data_path = os.path.join(data_dir, word_file)
        
        if not os.path.exists(data_path):
            print("perkele VITTO")
            return None
        
        word = dict()
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                key = line.split()[0].lower().strip()
                vec = np.array([float(x) for x in line.split()[1:]])
                word[key] = vec
                
        return word
    
    def _convert_data(self, data, word):
        
        s = next(iter(word.values())).shape
        result = []
        
        for d in data:
            words = d[0]
            label = d[1]
            vs = np.array([word.get(w, np.zeros(s)) for w in words])
            av = np.mean(vs, axis=0)
            
            result.append((av, label))
            
        return result
    
    ds_train = Family_Mart(data_dir, training_file, word_file)
    
    train_data_loader = torch.utils.data.DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    
    next(iter(train_data_loader))a
        
        
    
    
        
