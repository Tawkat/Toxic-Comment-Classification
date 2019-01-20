import pickle
from CNN.CNN_Create_Model import create_model_CNN,callback,plot_loss_accu
import numpy as np
from keras.models import load_model

pickle_load=open('Pickles/pickle_toxic_embedding_balanced_train_binary_100dim.pickle','rb')
X_train,y_train,embedding_matrix=pickle.load(pickle_load)
#print(y_train[10])

print(len(X_train),len(y_train))

pickle_load=open('Pickles/pickle_toxic_embedding_balanced_test_binary.pickle','rb')
X_test,y_test=pickle.load(pickle_load)

print(len(X_test),len(y_test))

vocabulary_size = embedding_matrix.shape[0]
dimension=300
embedding_size=100

# Convolution
filter_length = 3
nb_filter = 128

model_name = 'Models/CNN/Model_CNN_balanced_1.h5'
cb=callback(model_name=model_name)

model=create_model_CNN(dimension,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length,num_class=1)
history=model.fit(X_train,y_train,batch_size=512,epochs=10,validation_split=0.2,shuffle=True,callbacks=cb)

print("Saving Model...")
model_name = 'Models/CNN/Balanced/Model_CNN_balanced_1_class_2_sigmoid_binary_crossentropy.h5'#######################
model.save(model_name)

plot_loss_accu(history,lossLoc='Graph/CNN/Balanced/Train_Val_Loss_balanced_1_class_2_sigmoid_binary_crossentropy',
               accLoc='Graph/CNN/Balanced/Train_Val_acc_balanced_1_class_2_sigmoid_binary_crossentropy')


'''model_1=create_model_CNN(dimension,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length,num_class=1)
model_1.load_weights(model_name)
model_1.name = 'model_1' '''
#model=model_1
model=load_model('Models/CNN/Balanced/Model_CNN_balanced_1_class_2_sigmoid_binary_crossentropy.h5')

acc=model.evaluate(X_test,y_test)
print('Acc: '+str(acc[1]))

pred=model.predict(X_test)

'''t=4
xsd=X_test[0:5]
xsd=np.reshape(xsd,(-1,300))

#print(xsd.shape)

pred=model.predict(xsd)
print(pred)
print(y_test[0:5])'''
'''pred_class=model.predict_classes(X_test)
print(pred_class[0])'''

'''cnt=0
for i in range(len(y_test)):
    if(y_test[i].all()!=pred[i].all()):
        cnt+=1
        print(i)
        print(y_test[i])
        print(pred[i])
print('Inaccurate: '+str(cnt)+'/'+str(len(y_test)))'''

#pred_proba=model.predict_proba(X_test)

pred_binary=np.array(pred)
for i in range(len(pred_binary)):
    #print(pred_binary[i])
    pred_binary[i]=1*(pred_binary[i]>0.5)
    #print(pred_binary[i])

'''c=0
for i in range(len(y_test)):
    if(y_test[i]!=pred_binary[i]):
        print(i)
        c+=1
print(c/len(y_test))'''



print(pred[0])
print(pred_binary[0])
print(pred[1])
print(pred_binary[1])

from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import accuracy_score,jaccard_similarity_score

report=classification_report(y_test,pred_binary)
#report=precision_recall_fscore_support(y_test,pred_binary,average='micro')
print('Classification Report: '+str(report))


precision=precision_score(y_test,pred_binary,average='weighted')
print('Weighted Precision: '+str(precision))

recall=recall_score(y_test,pred_binary,average='weighted')
print('Weighted Recall: '+str(recall))

f1=f1_score(y_test,pred_binary,average='weighted')
print('Weighted F1 Score: '+str(f1))



acc_hard=accuracy_score(y_test,pred_binary)
print('Hard Accuracy: '+str(acc_hard))

from Custom_Metrics import accuracy_multilabel

acc_custom=accuracy_multilabel(y_test,pred_binary)
print('Custom Accuracy: '+str(acc_custom))


import pickle
with open('Predictions/CNN/Balanced/CNN_balanced_1_class_2_sigmoid_binary_crossentropy.pickle','wb') as f:
    pickle.dump((y_test,pred,pred_binary),f)

pickle_load='Predictions/CNN/Balanced/CNN_balanced_1_class_2_sigmoid_binary_crossentropy.pickle'
f=open(pickle_load,'rb')
y_test,pred,pred_binary=pickle.load(f)




#################################  Testing Section #########################

for i in range(len(y_test)):
    if(pred_binary[i].any()==0):
        print(pred_binary[i])

pred_binary[-10]=10
print(pred_binary[-10][0])

print(pred_binary[5],y_test[5])
