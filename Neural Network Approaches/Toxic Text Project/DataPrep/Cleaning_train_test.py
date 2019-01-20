import pandas as pd
from DataPrep.Clean_Texts import clean_text
import pickle

dataset=pd.read_csv('train_data.csv')
#dataset=dataset.head(10)

texts=dataset['comment_text']
#texts=texts.map(lambda x: clean_text(x))
for i in range(len(texts)):
    texts[i]=clean_text(texts[i])

label=dataset.iloc[:,2:8].values

#print(texts[0])

with open('Pickles/pickle_toxic_clean_train_Xy.pickle','wb') as f:
    pickle._dump((texts,label),f)

'''pickle_load=open('Pickles/pickle_toxic_clean_train_Xy.pickle','rb')
X,y=pickle.load(pickle_load)
print(X[0])'''


dataset_test=pd.read_csv('test_data.csv')

texts_test=dataset_test['comment_text']
#texts=texts.map(lambda x: clean_text(x))
for i in range(len(texts_test)):
    texts_test[i]=clean_text(texts_test[i])

label_test=dataset_test.iloc[:,2:8].values

#print(texts[0])

with open('Pickles/pickle_toxic_clean_test_Xy.pickle','wb') as f_test:
    pickle._dump((texts_test,label_test),f_test)