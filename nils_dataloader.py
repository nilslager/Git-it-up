import numpy as np
import pandas as pd
import csv
#Load the GloVe Embeddings

words = pd.read_table("/Users/nilslager/keio2018aia/course/general/project1/data/wv_50d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

#To get the vector for a word:

def vec(w):
  return words.loc[w].as_matrix()

#Test

print(vec("the"))
