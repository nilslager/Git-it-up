#Loading the GloVe file and making the "model" to readable form --> 50 dim. vector for each word
import numpy as np
f = open("/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/wv_50d.txt",'r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
print ("Done.",len(model)," words loaded!")

print (model['the'])
#open the reviews and split the sentences into words
words_found = 0

path='/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/senti_binary.train'
revfile=open(path,"r")
for line in revfile:
    splitrline=line.split()
    for i in splitrline:
        if i==1 or i==0:
            splitrline[i]=i
        else:
            try: 
                splitrline[i] = model[word]
                words_found += 1
            except KeyError:
                    splitrline[i] = np.random.normal(scale=0.6, size=(50, ))    
        
    print(splitrline)
    
matrix_len = len(model)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

#checking that the words are in vocabulary. Weights_matrix[] equals Model["  "]
for i, word in enumerate(model):
    try: 
        weights_matrix[i] = model[word]
        words_found += 1
    except word==1 or word == 0:
        weights_matrix[1]=word
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(50, ))

print('words found: ',words_found)
print(weights_matrix[2])

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
        
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class ToyNN(nn.Module):
    def __init__(self, weights_matrix, hidden_size, num_layers):
        super(self).__init__()
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        
    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
    

        
