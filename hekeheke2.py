import tensorflow as tf
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 14:38:33 2018

@author: henriparssinen
"""

import numpy as np
f = open("/Users/henriparssinen/desktop/Introduction to AI/keio2018aia/course/general/project1/data/wv_50d.txt",'r')
model = {}
for line in f:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
print ("Done.",len(model)," words loaded!")

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
print(ready_rev)       


names = ['neg','pos']   
texts,labels = [],[]
for idx,label in enumerate(names):
    for fname in glob.glob(os.path.join(f'{path}train', label, '*.*')):
        texts.append(open(fname, 'r').read())
        labels.append(idx)
trn,trn_y= texts, np.array(labels).astype(np.int64)


