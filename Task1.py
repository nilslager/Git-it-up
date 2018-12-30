# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:10:25 2018

@author: Ola
"""

###importing necesary packages
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
#Open file address and saving data to file
file = open('/Users/Ola/Documents/School/Keio/AI/wv_50d.txt' , 'r' , encoding='UTF-8')
#Define type for dictionary
dictionary = {}
#Begins loop, picking out single word's tied to 50 length vectors
for line in file:
    #Splits values into vector using split method
    splitLine = line.split()
    #Picks out word in vector
    word = splitLine[0]
    #Takes numbered values and turns them into 50 vector array of floating numbers
    embedding = np.array([float(val) for val in splitLine[1:]])
    dictionary[word] = embedding

#Create Array
#Set working path to training data
path='/Users/Ola/Documents/School/Keio/AI/senti_binary.train'
#Creates function returning workable data
def revfile_clean():
    #Defines data to be used in function
    revfile=open(path)
    #Creates new list
    clean_lines=list()
    #Starts for loop going through sentances string
    for line in revfile:
        #Split sentance string into single word strings
        splitrline=line.split()
        #Adds list of vectors to return target
        clean_lines.append(splitrline)
    #returns value to global variable
    return clean_lines

#Runs and puts return target from funtion revfile_clean into new variable
ready_rev=revfile_clean() 

####replace words with vectors

#Creates Dataloader class 
class Dataloader():
    #defines init  method
    def __init__(self, in_data):
        #
        self.in_data=in_data
        
    def extractor(self):
        #Starts for loop with values equal to number of sentances
        for i in range(len(self.in_data)):
            #Starts for loop with values equal to number of words in sentance
            for j in range(len(self.in_data[i])):
                if j is not (len(self.in_data[i])-1):
                    try:
                        self.in_data[i][j]=dictionary[self.in_data[i][j]]
                    except KeyError:
                        dictionary.update({self.in_data[i][j]:np.zeros(50)})
                        self.in_data[i][j]=dictionary[self.in_data[i][j]]
    
    def seperator_out(self):
        self.outputs=[]
        for sent in self.in_data:
            self.outputs.append(sent[-1])
            del sent[-1]
        self.outputs=np.array(self.outputs)
        self.outputs=self.outputs.astype(np.double)
        return(self.outputs)
##create a list of averaged sentences--> inputs
    def seperator_in(self):
        self.inputs=[]
        for sent in self.in_data:
            avgs=np.mean(sent,axis=0)
            self.inputs.append(avgs)
        return(self.inputs)


    #def out_test:

dataloader=Dataloader(ready_rev)
dataloader.extractor()

#Create dataset
outputs=dataloader.seperator_out()
inputs=dataloader.seperator_in()
data=list(zip(inputs,outputs))


###we can remove this I think
def sentence_to_mean_embeddings(review):
    for j in range(len(review)):
        if review[j] is not (len(review)-1):
            try:
                review[j]=dictionary[review[j]]
            except KeyError:
                dictionary.update({review[j]:np.zeros(50)})
                review[j]=dictionary[review[j]]
    avgs_test=torch.from_numpy(np.mean(review, axis=0))
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
    
    #Creates method train
    def train(self, lr, ne):
        
        #Defines pytorch function for finding mean square error as criterion
        criterion = nn.MSELoss()
        #Defines pytorch function for optimizing variables
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.1)#explore
        self.model.train()
        self.costs = []
        
        for e in range(ne):
            #Displays times neural network is trained
            print('training epoch %d / %d ...' %(e+1, ne))
            
            #Creates new value train_cost
            train_cost = 0
        
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                #Changes 50 vector array state to variables, 
                inputs = Variable(inputs)
                #Changes 0 or 1 rating state #check what variable does #maybe allows the two to not be connected
                targets = Variable(targets)
                #Clears the gradients of all optimized weights #### not sure if weights #### function description is tensors ####
                optimizer.zero_grad()
                #Saves calculated sentance reviews 
                outputs = self.model(inputs)
                #Calls function defined as criterion to find MSE
                loss = criterion(outputs, targets)
                train_cost += loss
                loss.backward()#research exact function of loss.backward
                
                optimizer.step()#research exact function of optimizer.step
                
            self.costs.append(train_cost/len(data))
            print('cost: %f' %(self.costs[-1]))

#defines class as trainer
trainer = Trainer(model, data)
#Train the data
trainer.train(0.005, 2)
plt.plot(range(len(trainer.costs)), trainer.costs)    

      
###Predictor
################################TEST##########################

path = '/Users/Ola/Documents/School/Keio/AI/senti_binary.test'
review=revfile_clean() 
predictor=neural_network_sentiment()

outputs=dataloader.seperator_out()
inputs=dataloader.seperator_in()
data=list(zip(inputs,outputs))


ac_outputs=[]
ac_cor=0
ac_tot=0


class Predictor():
    
    def __init__(self, model, data):
        
        self.model = model
        self.data = inputs
        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data, batch_size=10, shuffle=True)
        
    def pred(self, model, data):
        
        self.model.train()
        self.costs = []
        

        for sup in self.data:
        #Changes 50 vector array state #check what variable does
            inputs = Variable(torch.from_numpy(sup))
        #Saves calculated sentance reviews 
            outputs = self.model(inputs)
            return outputs
        
predictor=Predictor(model, data)

svar=predictor.pred(model, inputs)

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