import cython
import numpy as np
import bcolz
import pickle

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'/Users/nilslager/Desktop/gitit.50.dat', mode='w')

with open(f'/Users/nilslager/Desktop/wv_50d_gitit.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'/Users/nilslager/Desktop/gitit.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_index.pkl', 'wb'))

vectors = bcolz.open(f'/Users/nilslager/Desktop/6B.50.dat')[:]
words = pickle.load(open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_words.pkl', 'rb'))
word2idx = pickle.load(open(f'/Users/nilslager/Desktop/Projekt1/bibliotekN_index.pkl', 'rb'))
glove = {w: vectors[word2idx[w]] for w in words}
print(glove["the"])