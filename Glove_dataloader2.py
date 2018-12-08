
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:57:02 2018

@author: henriparssinen
"""
#download the Glove file
import numpy as np
f = open("/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/wv_50d.txt",'r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
print ("Done.",len(model)," words loaded!")

print (embedding[0:50])
print(model['sandberger'])

#Review material dataloader
from string import punctuation
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
import string
path='/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/senti_binary.train'

#Open the file
def doc_open(filename):
    revfile=open(path,"r")
    text=revfile.read()
    revfile.close()
    return text

#Clean for the vocabulary
def doc_clean(doc):
    splitrline=doc.split()
    table=str.maketrans("","",string.punctuation)
    splitrline=[w.translate(table) for w in splitrline]
    splitrline=[word for word in splitrline if word.isalpha()]
    stop_words=set(stopwords.words('english'))
    splitrline=[word for word in splitrline if not word in stop_words]
    splitrline=[word for word in splitrline if len(word)>1]
    return splitrline

path='/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/senti_binary.train'
text=doc_open(revfile)
splitrline=doc_clean(text)
print(splitrline) #this will give you a list of review file tokens(words)
print(len(splitrline))

# define vocab
vocab = Counter()
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
	# load doc
	doc = doc_open(filename)
	# clean doc
	splitrline = doc_clean(doc)
	# update counts
	vocab.update(splitrline)
    
add_doc_to_vocab(splitrline,vocab)    
print(len(vocab))
print (vocab)

# create a weight matrix for the Embedding layer from a loaded embedding(model in our case)
def get_weight_matrix(embed, vocab):
	# total vocabulary size
	vocab_size = len(vocab)
	# define weight matrix dimensions with all 0
	weight_matrix = np.zeros((vocab_size, 50))
	# step vocab, store vectors using the Tokenizer's integer mapping
	for word, i in vocab.items():
		vector = model.get(word)
		if vector is not None:
			weight_matrix[i] = vector
	return weight_matrix

matriisi=get_weight_matrix(model,vocab)
print(matriisi[1])
print(len(matriisi))
print(len(vocab))

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(splitrline)

print(tokenizer)

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(splitrline)
# pad sequences
max_length = max([len(s.split()) for s in splitrline])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')