import pandas as pd
from sklearn.utils import shuffle

'''df=pd.read_csv('train.csv')

label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df1=df.loc[(df["toxic"].astype(int)==1) | (df["severe_toxic"].astype(int)==1) | (df["obscene"].astype(int)==1)
           | (df["threat"].astype(int)==1) | (df["insult"].astype(int) == 1) | (df["identity_hate"].astype(int)==1)]

print(len(df1))

df0=df.loc[(df["toxic"].astype(int)==0) & (df["severe_toxic"].astype(int)==0) & (df["obscene"].astype(int)==0)
           & (df["threat"].astype(int)==0) & (df["insult"].astype(int) == 0) & (df["identity_hate"].astype(int)==0)]

print(len(df0))

df0=df0.sample(20000)
print(len(df0))

df_balanced=shuffle(pd.concat([df0,df1],ignore_index=True))
df_balanced.to_csv('balanced_toxic_data.csv',index=False)'''

df=pd.read_csv('balanced_toxic_data.csv')
df=shuffle(df)

#print(len(df))
#print(int(len(df)*0.8))
#print(int(len(df)*0.2))

train_size=int(len(df)*0.8)
test_size=int(len(df)*0.2)

df_train=df.head(train_size)
df_train.to_csv('balanced_train_data.csv',index=False)

df_test=df.tail(test_size)
df_test.to_csv('balanced_test_data.csv',index=False)



