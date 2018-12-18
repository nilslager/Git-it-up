import tensorflow as tf
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:38:33 2018
@author: OlaBandola
"""

import numpy as np
f = open('/Users/Ola/Documents/School/Keio/AI/wv_50d.txt','r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding

path='/Users/Ola/Documents/School/Keio/AI/senti_binary.train'
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

#1 replace words with vectors
for i in range(len(ready_rev)):
    for j in range(len(ready_rev[i])):
        if j is not (len(ready_rev[i])-1):
            if ready_rev[i][j+1] is "'s":
                ready_rev[i][j]=ready_rev[i][j]+ready_rev[i][j+1]
                #Delete j+1 word
                j=j-1
            try:
                ready_rev_edit[i][j]=np.sum(model[ready_rev[i][j]])
            except KeyError:
                model.update({ready_rev[i][j]:np.zeros(50)})
                ready_rev_edit[i][j]=model[ready_rev[i][j]]
                
            sum(np.sum(model[ready_rev[i][j]]).values(-j))
            np.sum(model(ready_rev_edit[1][1]))
            sum(model(ready_rev[1][1])).value())
        sum(model['character'].value())

                    
            
            
        
#Take the average of all words except for the last number in the file

names = ['neg','pos']   
texts,labels = [],[]
for idx,label in enumerate(names):
    for fname in glob.glob(os.path.join(f'{path}train', label, '*.*')):
        texts.append(open(fname, 'r').read())
        labels.append(idx)
trn,trn_y= texts, np.array(labels).astype(np.int64)