import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv('train.csv')

df=shuffle(df)

#print(len(df))
#print(int(len(df)*0.8))
#print(int(len(df)*0.2))

train_size=int(len(df)*0.8)
test_size=int(len(df)*0.2)

df_train=df.head(train_size)
df_train.to_csv('train_data.csv',index=False)

df_test=df.tail(test_size)
df_test.to_csv('test_data.csv',index=False)



