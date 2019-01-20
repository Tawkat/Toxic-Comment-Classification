from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM,Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras import optimizers
from keras.layers import TimeDistributed
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder

vocabulary_size = 400000
#***********
time_step=300
embedding_size=100

pickle_train=open('Pickles/pickle_toxic_clean_train_Xy.pickle','rb')
texts,y_train=pickle.load(pickle_train)

tokenizer_train=Tokenizer(num_words=vocabulary_size)
tokenizer_train.fit_on_texts(texts)
encoded_train=tokenizer_train.texts_to_sequences(texts=texts)
#print(encoded_docs)
vocab_size_train = len(tokenizer_train.word_index) + 1
print(vocab_size_train)

X_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')


f = open('glove.6B.100d.txt',encoding='utf-8')
embeddings_train={}
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_train[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_train))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size_train, embedding_size))
for word, i in tokenizer_train.word_index.items():
	embedding_vector_train = embeddings_train.get(word)
	if embedding_vector_train is not None:
		embedding_matrix[i] = embedding_vector_train


with open('Pickles/pickle_toxic_embedding_train_100dim.pickle','wb') as f:
    pickle.dump((X_train,y_train,embedding_matrix),f)




#######################   test   ##############################################

pickle_test=open('Pickles/pickle_toxic_clean_test_Xy.pickle','rb')
texts_test,y_test=pickle.load(pickle_test)

encoded_test=tokenizer_train.texts_to_sequences(texts=texts_test)
X_test = sequence.pad_sequences(encoded_test, maxlen=time_step, padding='post')

with open('Pickles/pickle_toxic_embedding_test.pickle','wb') as f_test:
    pickle.dump((X_test,y_test),f_test)