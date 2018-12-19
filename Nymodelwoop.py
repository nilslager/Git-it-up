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
#%matplotlib inline
from pylab import rcParams

f = open('/Users/nilslager/Desktop/Projekt1/wv_50d.txt','r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding

path='/Users/nilslager/Desktop/senti_binary.train'
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
input_average=[]
output_sentiment=[]


# Update dictionary
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            try:
                model[ready_rev[i][j]]
            except KeyError:
                model.update({ready_rev[i][j]:np.zeros(50)})


#1 replace words with vectors
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            sum_ready_rev=sum_ready_rev+model[ready_rev[i][j]]
        else:
            sum_ready_rev=sum_ready_rev/(len(ready_rev[i])-1)
            output_sentiment.append(ready_rev[i][-1])
    input_average.append(sum_ready_rev)
    
#Converting lists to Tensors
input_average_n=np.array(input_average)
output_sentiment_n=np.array(output_sentiment)
ny=np.column_stack((input_average_n, output_sentiment_n))
input_average_n=input_average_n.astype(np.float32)
input_average_nn=torch.from_numpy(input_average_n)
output_sentiment_n=output_sentiment_n.astype(np.float32)
output_sentiment_nn=torch.from_numpy(output_sentiment_n)
ny_n=ny.astype(np.float32)
ny_nn=torch.from_numpy(ny_n)

type(ny_nn)
            
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


model = Model(50, 100, 20, 1)
model.double()
#Fuck-shit-up
# Create trainer
class Trainer():
    
    def __init__(self, model, ny_nn):
        
        self.model = model
        self.ny_nn = ny_nn
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.ny_nn, batch_size=100, shuffle=True)
        
    def train(self, lr, ne):
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)

        self.model.train()
        
        self.costs = []
        
        for e in range(ne):
            
            print('training epoch %d / %d ...' %(e+1, ne))
            
            train_cost = 0
        
            for batch_idx, (input_average_nn, targets) in enumerate(self.train_loader):

                input_average_nn = Variable(input_average_nn)
                targets = Variable(targets)

                optimizer.zero_grad()
                output_sentiment_nn = self.model(input_average_nn)
                loss = criterion(output_sentiment_nn, targets)
                train_cost += loss
                loss.backward()
                optimizer.step()
                
            self.costs.append(train_cost/len(ny_nn))
            print('cost: %f' %(self.costs[-1]))

trainer = Trainer(model, ny_nn)
#learning

trainer.train(0.1, 100)

plt.plot(range(len(trainer.costs)), trainer.costs)
#Nytt
