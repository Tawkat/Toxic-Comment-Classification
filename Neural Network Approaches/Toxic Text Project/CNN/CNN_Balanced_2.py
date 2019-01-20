import pickle
from CNN.CNN_Create_Model import create_model_CNN,callback,plot_loss_accu
import numpy as np

pickle_load=open('Pickles/pickle_toxic_embedding_balanced_train_100dim.pickle','rb')
X_train,y_train,embedding_matrix=pickle.load(pickle_load)

y_temp_train=np.zeros(shape=(y_train.shape[0],y_train.shape[1]+1))

for i in range(len(y_train)):
    if(y_train[i].any()==0):
        y_temp_train[i,0]=1
    y_temp_train[i,1:]=y_train[i]

y_train=np.array(y_temp_train)
#print(y_train[6])

pickle_load=open('Pickles/pickle_toxic_embedding_balanced_test.pickle','rb')
X_test,y_test=pickle.load(pickle_load)

y_temp_test=np.zeros(shape=(y_test.shape[0],y_test.shape[1]+1))

for i in range(len(y_test)):
    if(y_test[i].any()==0):
        y_temp_test[i,0]=1
    y_temp_test[i,1:]=y_test[i]

y_test=np.array(y_temp_test)

vocabulary_size = embedding_matrix.shape[0]
timeStep=300
embedding_size=100

# Convolution
filter_length = 3
nb_filter = 128

model_name = 'Models/CNN/Model_CNN_balanced_1.h5'
cb=callback(model_name=model_name)

model=create_model_CNN(timeStep,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length)
history=model.fit(X_train,y_train,batch_size=512,epochs=10,validation_split=0.2,shuffle=True,callbacks=cb)

print("Saving Model...")
model_name = 'Models/CNN/Model_CNN_balanced_1.h5'########################################3
model.save(model_name)

plot_loss_accu(history,lossLoc='Graph/CNN/Balanced/Train_Val_Loss',accLoc='Graph/CNN/Balanced/Train_Val_acc')


model_1=create_model_CNN(timeStep,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length)
model_1.load_weights('Models/CNN/Model_CNN_balanced_1.h5')
model_1.name = 'model_1'


acc=model.evaluate(X_test,y_test)
print('Acc: '+str(acc[1]))

pred=model.predict(X_test)
#print(pred[0])
#print(y_test[10])
'''pred_class=model.predict_classes(X_test)
print(pred_class[0])'''

'''cnt=0
for i in range(len(y_test)):
    if(y_test[i].any()!=pred_binary[i].any()):
        cnt+=1
        print(i)
        print(y_test[i])
        print(pred_binary[i])
print('Inaccurate: '+str(cnt)+'/'+str(len(y_test)))'''

#pred_proba=model.predict_proba(X_test)

pred_binary=np.array(pred)
for i in range(len(pred_binary)):
    print(pred_binary[i])
    pred_binary[i]=1*(pred_binary[i]>0.5)
    print(pred_binary[i])

print(pred[0])
print(pred_binary[0])
print(pred[3507])
print(pred_binary[3507])

from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score
report=classification_report(y_test,pred_binary)
#report=precision_recall_fscore_support(y_test,pred_binary,average='micro')
print(report)

acc_report=accuracy_score(y_test,pred_binary)
print(acc_report)




#################################  Testing Section #########################

for i in range(len(y_test)):
    if(y_test[i].any()==0):
        print(y_test[i])

pred_binary[-10]=10
print(pred_binary[-10])