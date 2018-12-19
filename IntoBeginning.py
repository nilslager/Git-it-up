import tensorflow as tf
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:38:33 2018
@author: OlaBandola
"""

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from pylab import rcParams

f = open('/Users/Ola/Documents/School/Keio/AI/wv_50d.txt','r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding

path='/Users/Ola/Documents/School/Keio/AI/senti_binary.train'
revfile=open(path)

def revfile_clean():
    revfile=open(path)
    clean_lines=list()
    for line in revfile:
        splitrline=line.split()
        clean_lines.append(splitrline)
    return clean_lines

ready_rev=revfile_clean() 
ready_rev_edit=revfile_clean()      
sum_ready_rev=np.zeros(50)
total_sum_ready_rev=[]
indicator=[]


# Update dictionary
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            try:
                model[ready_rev[i][j]]
            except KeyError:
                model.update({ready_rev[i][j]:np.zeros(50)})


# Model creatation
                
# Creat Network
class Model(nn.Module):
    
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        
        super(Model, self).__init__()
        
        self.hl1 = nn.Linear(input_dim, hidden1_dim)
        self.hl1a = nn.ReLU()
        self.layer1 = [self.hl1, self.hl1a]
        
        self.hl2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.hl2a = nn.ReLU()
        self.layer2 = [self.hl2, self.hl2a]
        
        self.ol = nn.Linear(hidden2_dim, output_dim)
        self.ola = (lambda x: x)
        self.layer3 = [self.ol, self.ola]
        
        self.layers = [self.layer1, self.layer2, self.layer3]
        
    def forward(self, x):
        
        out = x
        
        for pa, a in self.layers:
            
            out = a(pa(out))
        
        return out



# Create trainer
class Trainer():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = data
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=8, shuffle=True)
        
    def train(self, lr, ne):
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        self.model.train()
        
        self.costs = []
        
        for e in range(ne):
            
            print('training epoch %d / %d ...' %(e+1, ne))
            
            train_cost = 0
        
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):

                inputs = Variable(inputs)
                targets = Variable(targets)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                train_cost += loss
                loss.backward()
                optimizer.step()
                
            self.costs.append(train_cost/len(data))
            print('cost: %f' %(self.costs[-1]))



#1 replace words with vectors
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            sum_ready_rev=sum_ready_rev+model[ready_rev[i][j]]
        else:
            sum_ready_rev=sum_ready_rev/(len(ready_rev[i])-1)
            indicator.append(ready_rev[i][-1])
    total_sum_ready_rev=total_sum_ready_rev+sum_ready_rev
    
    #inster class function here


names = ['neg','pos']   
texts,labels = [],[]
for idx,label in enumerate(names):
    for fname in glob.glob(os.path.join(f'{path}train', label, '*.*')):
        texts.append(open(fname, 'r').read())
        labels.append(idx)
trn,trn_y= texts, np.array(labels).astype(np.int64)