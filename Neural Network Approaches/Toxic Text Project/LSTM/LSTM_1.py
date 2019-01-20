from Clean_Texts import clean_text
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import pickle

embedding_size=100


pickle_load=open('Pickles/pickle_toxic_embedding_train_100dim.pickle','rb')
X_train,y_train,embedding_matrix=pickle.load(pickle_load)

pickle_load=open('Pickles/pickle_toxic_embedding_test.pickle','rb')
X_test,y_test=pickle.load(pickle_load)



from LSTM.LSTM_Create_Model import create_model_LSTM

vocabulary_size=embedding_matrix.shape[0]
model=create_model_LSTM(vocabulary_size,embedding_size,embedding_matrix)
history=model.fit(X_train,y_train,batch_size=512,epochs=5,validation_split=0.2,shuffle=True)

print("Saving Model...")
model_name = 'Models/LSTM/Model_LSTM_1.h5'########################################3
model.save(model_name)


model_1=create_model_LSTM(vocabulary_size,embedding_size,embedding_matrix)
model_1.load_weights('Models/LSTM/Model_LSTM_1.h5')
model_1.name = 'model_1'


acc=model.evaluate(X_test,y_test)
print('Acc: '+str(acc[1]))

pred=model.predict(X_test)
print(pred[10])



'''df2=dataset.head(10)
print(df2)
df3=dataset.tail(990)
print(df3)
print(dataset.shape)'''

#print(texts[0])

'''pred=model.predict_proba([X_train[0],X_train[1]])
print(len(pred))
print(pred[5])'''