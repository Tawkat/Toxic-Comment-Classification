from Clean_Texts import clean_text
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

vocabulary_size = 400000
#***********
time_step=300
embedding_size=100

dataset=pd.read_csv('balanced_toxic_data.csv')
#dataset=dataset.head(1000)

texts=dataset['comment_text']
texts=texts.map(lambda x: clean_text(x))

label=dataset.iloc[:,2:8].values
#print(label)


tokenizer_train=Tokenizer(num_words=vocabulary_size)
tokenizer_train.fit_on_texts(texts)
encoded_train=tokenizer_train.texts_to_sequences(texts=texts)
#print(encoded_docs)
vocab_size_train = len(tokenizer_train.word_index) + 1
print(vocab_size_train)

X_train = sequence.pad_sequences(encoded_train, maxlen=time_step,padding='post')
y_train=label


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



from LSTM.LSTM_Create_Model import create_model_LSTM

vocabulary_size=embedding_matrix.shape[0]
model=create_model_LSTM(vocabulary_size,embedding_size,embedding_matrix)
history=model.fit(X_train,y_train,batch_size=512,epochs=5,validation_split=0.2,shuffle=True)


'''df2=dataset.head(10)
print(df2)
df3=dataset.tail(990)
print(df3)
print(dataset.shape)'''

#print(texts[0])

'''pred=model.predict_proba([X_train[0],X_train[1]])
print(len(pred))
print(pred[5])'''