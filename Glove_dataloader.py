
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