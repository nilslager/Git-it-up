import torch
import torch.nn as nn
import matplotlib.pyplot as plt
%matplotlib inline

from torch.autograd import Variable

from data import iris

glovedict = open("/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt", "r")
words = []
index = 0
word2index = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt', mode='w')

with open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2index[word] = index
        index += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)
    
vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt', mode='w')
vectors.flush()
pickle.dump(words, open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.pkl', 'wb'))
pickle.dump(word2idx, open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.pkl', 'wb'))


vectors = bcolz.open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt')[:]
words = pickle.load(open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.pkl', 'rb'))
word2index = pickle.load(open(f'/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt', 'rb'))

glove = {w: vectors[word2index[w]] for w in words}