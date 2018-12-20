
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:38:33 2018
@author: OlaBandola
"""
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
#%matplotlib inline
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import numpy as np

#Open embedding
f = open("/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/wv_50d.txt",'r')
dictionary = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    dictionary[word] = embedding


path='/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/senti_binary.train'
revfile=open(path)

def revfile_clean():
    revfile=open(path)
    clean_lines=list()
    for line in revfile:
        splitrline=line.split()
        clean_lines.append(splitrline)
    return clean_lines

ready_rev=revfile_clean() 
ready_rev_edit=ready_rev      

#1 replace words with vectors
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            if ready_rev[i][j+1] is "'s":
                ready_rev[i][j]=ready_rev[i][j]+ready_rev[i][j+1]
                j=j-1
            try:
                ready_rev_edit[i][j]=dictionary[ready_rev[i][j]]
            except KeyError:
                dictionary.update({ready_rev[i][j]:np.zeros(50)})
                ready_rev_edit[i][j]=dictionary[ready_rev[i][j]]

#create a list of neg or pos reviews --> outputs                
outputs=[]
for sent in ready_rev_edit:
    outputs.append(sent[-1])
    del sent[-1]
outputs=np.array(outputs)
outputs=outputs.astype(np.double)
##create a list of averaged sentences--> inputs
inputs=[]
for sent in ready_rev_edit:
    avgs=np.mean(sent,axis=0)
    inputs.append(avgs)
  
#combining lists-->list of tuples
data=list(zip(inputs,outputs))
print(type(inputs[0]))
print(type(outputs[0]))



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


class Trainer():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = data

        self.train_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=10, shuffle=True)
    
        
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
            
trainer = Trainer(model, data)

trainer.train(0.005, 10)
plt.plot(range(len(trainer.costs)), trainer.costs)

