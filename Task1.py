# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:10:25 2018

@author: Ola
"""

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
import numpy as np

#Create Dictionary
file = open('/Users/Ola/Documents/School/Keio/AI/wv_50d.txt' , 'r' , encoding='UTF-8') #may need encoding
dictionary = {}
for line in file:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    dictionary[word] = embedding

#Create Array 
path='/Users/Ola/Documents/School/Keio/AI/senti_binary.train'
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

#Class clean(datatype, data)

    #def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):

    #def clean:
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            try:
                ready_rev_edit[i][j]=dictionary[ready_rev[i][j]]
            except KeyError:
                dictionary.update({ready_rev[i][j]:np.zeros(50)})
                ready_rev_edit[i][j]=dictionary[ready_rev[i][j]]
    
    #def out_train:
    
    #def out_test:
        
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

#Create dataset
data=list(zip(inputs,outputs))

def text_preprocess(sentence):
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    words = sentence_text.lower().split()
    return (words)



def sentence_to_mean_embeddings(review):
    for j in range(len(review)):
        if review[j] is not (len(review)-1):
            try:
                review[j]=dictionary[review[j]]
            except KeyError:
                dictionary.update({review[j]:np.zeros(50)})
                review[j]=dictionary[review[j]]
    for sent in review:
        avgs_test=np.mean(sent,axis=0)
    return (avgs_test)




    

###Create model

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

###Create Trainer
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

#Train the data
trainer.train(0.005, 100)
plt.plot(range(len(trainer.costs)), trainer.costs)    

      
###Predictor
class neural_network_sentiment(nn.Module):
    def __init__(self):
        super().__init__()
        
    def predict_prob(self,review):
        prob_distribution=F.softmax(self.forward(sentence_to_mean_embeddings(review)),dim=0)
        return prob_distribution.data
    
    def predict(self,review):
        return int(torch.max(self.forward(sentence_to_mean_embeddings(review)), 0)[1].data.numpy())
   
def Predictor(review):
    if neural_network_sentiment.predict()==0:
        return("This is a positive review")
    else:
        return("This is a negative review")
        

#print(Predictor("Horrible movie"))    
    
#    def fit(self,train_loader,test_loader,initial_lr=0.01,gamma=0.95,n_epochs=20, score=False,print_results=True,chart=True)
    

###Accuracy for predictor        
path = '/Users/Ola/Documents/School/Keio/AI/senti_binary.test'
review=revfile_clean() 
predictor=neural_network_sentiment()
ac_outputs=[]
ac_cor=0
ac_tot=0
#sentence_to_mean_embeddings()

ac_outputs=[]
for sent in review:
    ac_outputs.append(sent[-1])
    del sent[-1]
for i in review:
    ratings1 = predictor.predict_prob(i)
    ratings2 = predictor.predict(i)
    if ac_outputs[ac_tot] == ratings2:
        ac_cor=ac_cor+1
        ac_tot=ac_tot+1
    else:
        ac_tot=ac_tot+1
print('accuracy is ' + str(ac_cor/ac_tot) + '.')

    

#plt.scatter(inputs, outputs, c='k', marker='o', s=0.1)
#plt.scatter(inputs, model(torch.tensor(inputs)).detach().numpy().flatten(), c='r', marker='o', s=0.1)