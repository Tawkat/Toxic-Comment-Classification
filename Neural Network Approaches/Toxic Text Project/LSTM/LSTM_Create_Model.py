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
from keras import callbacks

def create_model_LSTM(vocabulary_size,embedding_size,embedding_matrix,num_class=6):
    ## create model
    model = Sequential()
    model.add(Embedding(vocabulary_size, embedding_size, weights=[embedding_matrix],trainable=False))

    model.add(LSTM(100))

    #model.add(Dense(units=64))  #############################
    # model.add(Dropout(rate=0.8)) #############################

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()
    return model


def callback(model_name,tf_log_dir_name='./tf-log/',patience_lr=2,):
    cb = []
    """
    Tensorboard log callback
    """
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb.append(tb)

    """
    Model-Checkpoint
    """
    m = callbacks.ModelCheckpoint(filepath=model_name,monitor='val_loss',mode='auto',save_best_only=True)
    cb.append(m)

    """
    Reduce Learning Rate
    """
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    cb.append(reduce_lr_loss)

    """
    Early Stopping callback
    """
    # Uncomment for usage
    early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')
    cb.append(early_stop)



    return cb

######### Show Train Val History Graph ###############
def plot_loss_accu(history,lossLoc='Train_Val_Loss',accLoc='Train_Val_acc'):
    import matplotlib.pyplot as plt

    plt.clf()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.title('Training and validation loss')
    plt.legend(['train', 'val'], loc='upper right')
    #plt.show()
    plt.savefig(lossLoc)

    plt.clf()

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.title('Training and validation accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    #plt.show()
    plt.savefig(accLoc)