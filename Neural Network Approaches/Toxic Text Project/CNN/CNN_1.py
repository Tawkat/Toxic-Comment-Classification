import pickle
from CNN.CNN_Create_Model import create_model_CNN

pickle_load=open('Pickles/pickle_toxic_embedding_train_100dim.pickle','rb')
X_train,y_train,embedding_matrix=pickle.load(pickle_load)

pickle_load=open('Pickles/pickle_toxic_embedding_test.pickle','rb')
X_test,y_test=pickle.load(pickle_load)

vocabulary_size = embedding_matrix.shape[0]
timeStep=300
embedding_size=100

# Convolution
filter_length = 3
nb_filter = 128

model=create_model_CNN(timeStep,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length)
#history=model.fit(X,y,epochs=10,batch_size=512,shuffle=True)
history=model.fit(X_train,y_train,batch_size=512,epochs=10,validation_split=0.2,shuffle=True)

print("Saving Model...")
model_name = 'Models/CNN/Model_CNN_1.h5'########################################3
model.save(model_name)


model_1=create_model_CNN(timeStep,vocabulary_size,embedding_size,embedding_matrix,nb_filter,filter_length)
model_1.load_weights('Models/CNN/Model_CNN_1.h5')
model_1.name = 'model_1'


acc=model.evaluate(X_test,y_test)
print('Acc: '+str(acc[1]))

pred=model.predict(X_test)
print(pred[10])