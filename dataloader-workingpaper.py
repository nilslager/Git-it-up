#Data loader
import cython
import numpy as np
import bcolz
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

#Dataloader
words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'/Users/nilslager/Desktop/gitit.50.dat', mode='w')

#Open up GloVe embeddings and create vectors
with open(f'/Users/nilslager/Desktop/wv_50d_gitit.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
        
#Construct pickle files
vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'/Users/nilslager/Desktop/gitit.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_index.pkl', 'wb'))

#Create vector space
vectors = bcolz.open(f'/Users/nilslager/Desktop/6B.50.dat')[:]
words = pickle.load(open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_words.pkl', 'rb'))
word2idx = pickle.load(open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_index.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}

#Test
print(glove["the"])

#Model
matrix_len = len(glove)
weights_matrix = np.zeros((matrix_len, 50))
words_found = 0

#Glove -> Weights matrix
for i, word in enumerate(glove):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(50))

#Create NN - First layer is embedding layer and GRU layer.
def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

#Create NN class
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