#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 18:06:40 2018

@author: nilslager
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
import re

#Open embedding
#f = open('/Users/nilslager/Desktop/keio2018aia/course/general/project1/data/wv_50d.txt','r')
#dictionary = {}
#for line in f:
#    splitLine = line.split()
 #   word = splitLine[0]
  #  embedding = np.array([float(val) for val in splitLine[1:]])
   # dictionary[word] = embedding

path='/Users/nilslager/Downloads/Skola/Keio/keio2018aia/course/general/project1/data/senti_binary.train'
revfile=open(path)

def revfile_clean():
    revfile=open(path)
    clean_lines=list()
    for line in revfile:
        splitrline=line.split()
        clean_lines.append(splitrline)
    return clean_lines

ready_rev=revfile_clean() 
train_list=ready_rev

##############################################################################
#Part II
#############################################################################
#Define the model we are going to train

#index nr for each unique word
#with glove 50dim:
#now 15 dimensional vector randomized
#new model
#Some function
def text_preprocess(sentence):
    
    #1st step: Lowercase
#    sentence_lower =sentence.lower()
    #2nd step: replace '-' with ' ' so we won't lose words in step 5
#    sentence_replace=sentence_lower.replace('-',' ')
  
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    # 2. Convert words to lower case and split them
    words = sentence_text.lower().split()
    
    #3rd step: tokenization
    #sentence_token = word_tokenize(sentence, replace)
    
    #4th step: apostrophe handling
#    sentence_apostrophe = [apostrophe_dict[word] if word in apostrophe_dict else word for word in sentence_token]
    

    #6th step: Removing non alphabetic characters
 #   sentence_alpha = [word for word in sentence_apostrophe if word.isalpha()]
    
    #7th step: Removing words we don't know
  #  sentence_known = [word for word in sentence_alpha if word in word_embeddings]
    
    return (words)
    
#############################################################################
#Build a new dictionary of words

#First, preprocess all sentences in the train set
train_sentences_list=[text_preprocess(datapoint[0]) for datapoint in train_list]

#Getting the length of the sentences (it will be useful later)
train_sentences_lengths=np.array([len(sentence) for sentence in train_sentences_list])

#Now, flatten the list, remove the duplications and sort the list
train_words_list = [word for sentence in train_sentences_list for word in sentence]

train_unique_words_list = []
for word in train_words_list:
    if not (word in train_unique_words_list):
        train_unique_words_list.append(word)
    
train_unique_words_list.sort()

#Finally, let's create the dictionary which associates
word_to_idx = {word:(1+i) for i, word in enumerate(train_unique_words_list)}
word_to_idx['__unknown__']=0
idx_to_word={word_to_idx[word]: word for word in word_to_idx}
print('The length of our dictionary is: ', len(word_to_idx))

idx_to_word[4879]



#create _unkown_, give value 0.
class deep_nn():
    
    def __init__(self, dictionary_length, n_embeddings, n_hidden, n_classes):
#        super()._init_()
        self.dictionary_length=dictionary_length
        self.n_embeddings=n_embeddings
        self.n_classes=n_classes
        self.n_hidden=n_hidden
        
        self.act=nn.ReLU()
        self.embeddings=nn.Embedding(dictionary_length, self.n_embeddings, padding_idx=0, sparse=True)
        self.fc1=nn.Linear(n_embeddings,n_hidden)
        self.fc2=nn.Linear(n_hidden, n_classes)
        
        
    def forward(self,x):
        embeds=self.embeddings(x)
        
        if len(embeds.shape) == 2:
            mean_embeds=torch.mean(embeds,0)
        
        elif len(embeds.shape) ==3:
            mean_embeds = []
            for embed in embeds:
                mean_embeds.append(torch.mean(embed,0).reshape(1,self.n_embeddings))
                mean_embeds=torch.cat(mean_embeds,0)
    
        out=self.fc1(mean_embeds)
        out=self.act(out)
        out=self.fc2(out)
        return(out)

#Instantiate
my_deep_nn_0_0= deep_nn(len(word_to_idx),15,30,2)

#Train
#my_deep_nn_0_0.fit(train_loader)

# Compute embeddings and spot trends in their squared norms

#Create a dictionary with our learned embeddings
my_embeddings={word: my_deep_nn_0_0.embeddings(torch.LongTensor([word_to_idx[word]])) for word in word_to_idx}

new_ready_rev=revfile_clean() 
new_ready_rev_edit=new_ready_rev  

#1 replace words with vectors
for i in range(len(new_ready_rev)):
    for j in range(len(new_ready_rev[i])):
        if j is not (len(new_ready_rev[i])-1):
            if new_ready_rev[i][j+1] is "'s":
                new_ready_rev[i][j]=new_ready_rev[i][j]+new_ready_rev[i][j+1]
                j=j-1
            try:
                new_ready_rev_edit[i][j]=my_embeddings[new_ready_rev[i][j]]
            except KeyError:
                my_embeddings.update({new_ready_rev[i][j]:np.zeros(15)})
                new_ready_rev_edit[i][j]=my_embeddings[new_ready_rev[i][j]]

new_ready_rev_edit
#create a list of neg or pos reviews --> outputs                
outputs=[]
for sent in new_ready_rev_edit:
    outputs.append(sent[-1])
    del sent[-1]
outputs=np.array(outputs)
outputs=outputs.astype(np.double)
##create a list of averaged sentences--> inputs
inputs=[]
for sent in new_ready_rev_edit:
    avgs=np.mean(sent,axis=0)
    inputs.append(avgs)

#Create dataset
data=list(zip(inputs,outputs))



len(my_embeddings)

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
    
model = Model(15, 100, 20, 1)
model.double()

#Create Trainer
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

##############################################################################
#Part III
#############################################################################
class perceptron():
    
    def __init__(self, dictionary_length, n_embeddings, n_classes):
        super().__init__()
        self.dictionary_length = dictionary_length
        self.n_embeddings=n_embeddings = n_embeddings
        #Initialize the weighting by uniform (0,1) to random numbers
        self.attention_vector=torch.rand(n_embeddings).float().reshape(n_embeddings,1).to(device) 
        self.n_classes=n_classes


        #padding_idx=0 makes sure the embedding for the __unknown__ word is 0
        self.embeddings=nn.Embedding(dictionary_length,self.n_embeddings,padding_idx=0,sparse=True)
        self.fc1=nn.Linear(self.n_embeddings, self.n_classes)

    def attention_layer(self,embeds):
        if len(embeds.shape) == 2: #If we feed the network a single example
            weights_non_normalized =torch.mm(embeds,self.attention_vector).reshape(1,-1)
            weights_normalized=F.softmax(weights_non_normalized, dim=1)
        
            #compute the average embedding of each sentence
            mean_embeds=torch.mm(weights_normalized,emebds).reshape(-1)
        
        elif len(embeds.shape) == 3: #If we feed the network a batch of examples
            #Retrieve the dimensino of a batch
            batch_size=embeds.shape[0]
        
            mean_embeds=[]
            for i in range(batch_size):
                weights_non_normalized = torch.mm(embeds[i], self.attention_vector).reshape(-1,1)
                weights_normalized = F.softmax(weights_non_normalized, dim=1)
                #compute weighted average embedding of the sentence
                mean_embeds.append(torch.mm(weights_normalized, embeds[i].reshape(-1,1)))
            
            mean_embeds = torch.cat(mean_embeds,0)
            
        return mean_embeds
    
    def forward(self,x):
        
        #Embedding layer
        embeds = self.embeddings(x)
        
        #Weighting Layer
        mean_embeds=self.attention_layer(embeds)
        
        #Linear Layer
        out = self.fc1(mean_embeds)
        
        return(out)
        
#Instantiate the perceptron OBS NOTE ONLY ONE D???
my_embedings_perceptron = perceptron(len(word_to_idx),10,2)
#Train the perceptron (uncomment)
my_embedings_perceptron.fit(train_loader,test_loader,gpu=gpu)
torch.save(my_embedings_perceptron.state_dict(),"./part3_state.chkpt")