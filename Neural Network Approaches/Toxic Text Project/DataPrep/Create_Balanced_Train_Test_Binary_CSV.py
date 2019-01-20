import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv('balanced_test_data.csv')
print(len(df))

label_list = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

'''df1=df.loc[(df["toxic"].astype(int)==1) | (df["severe_toxic"].astype(int)==1) | (df["obscene"].astype(int)==1)
           | (df["threat"].astype(int)==1) | (df["insult"].astype(int) == 1) | (df["identity_hate"].astype(int)==1)]
label=[]
for i in range(len(df1['toxic'])):
    label.append(1)

print(len(df1))

df0=df.loc[(df["toxic"].astype(int)==0) & (df["severe_toxic"].astype(int)==0) & (df["obscene"].astype(int)==0)
           & (df["threat"].astype(int)==0) & (df["insult"].astype(int) == 0) & (df["identity_hate"].astype(int)==0)]
for i in range(len(df0['toxic'])):
    label.append(0)

print(len(df0))'''

label=[]

for i in range(len(df)):
    toxic=False
    for l in label_list:
        if(df[l][i].astype(int)==1):
            label.append(1)
            toxic=True
            break
    if(toxic==False):
        label.append(0)


'''dfc1=df1[['id','comment_text']]
dfc0=df0[['id','comment_text']]'''

df2=df[['id','comment_text']]
df2['label']=label

#df2=pd.concat([dfc1,dfc0],ignore_index=True)
#df2['label']=label
print(len(df),len(label),len(df2))

df_balanced_binary=pd.DataFrame(df2)
df_balanced_binary.to_csv('balanced_test_binary_data.csv',index=False)

df=pd.read_csv('balanced_toxic_binary_data.csv')
df=shuffle(df)

#print(len(df))
#print(int(len(df)*0.8))
#print(int(len(df)*0.2))

train_size=int(len(df)*0.8)
test_size=int(len(df)*0.2)

df_train=df.head(train_size)
df_train.to_csv('balanced_train_binary_data.csv',index=False)

df_test=df.tail(test_size)
df_test.to_csv('balanced_test_binary_data.csv',index=False)



