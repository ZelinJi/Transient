#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import theano as th
# import theano.tensor as T
# from keras.utils import np_utils
# import keras.models as models
# from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
# from keras.layers.noise import GaussianNoise
# from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
# from keras.regularizers import *
# from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# import seaborn as sns
# import random, sys, keras
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[36]:


for dirname, _, filenames in os.walk('./dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[37]:


xd = pickle.load(open(r"./dataset/RML2016.10a_dict.dat", 'rb'),encoding='iso-8859-1')


# In[38]:


snrs,mods = [sorted(list(set([x[j] for x in list(xd.keys())]))) for j in [1,0]]
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(xd[(mod,snr)])
        for i in range(xd[(mod,snr)].shape[0]):  
            lbl.append((mod,snr))
X = np.vstack(X)


# In[39]:


np.random.seed(2016)
n_examples = X.shape[0]
n_train = int(n_examples * 0.8)

train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


# In[40]:


# building thie model using tensorflow library (sequencial model)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Reshape
from tensorflow.keras.layers import Convolution2D
import tensorflow.keras.models as models

dr = 0.2 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp + [1], input_shape=in_shp))
model.add(Convolution2D(50, (1, 8), padding='same', activation="relu", name="conv1", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Convolution2D(50, (2, 8), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
# model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy','mse','mae','mape'])
model.summary()


# In[41]:


nb_epoch = 100
batch_size = 1024

# training the model
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=1,
    validation_data=(X_test, Y_test),
    shuffle=True,
    class_weight=None,
    )

# saving the model
model.save('model.h5')

# loading the model
from tensorflow.keras.models import load_model
model = load_model('model.h5')

# testing the model
score = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)


# In[42]:


# plotting the accuracy and loss
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Accuracy')

plt.subplot(4, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Loss')

plt.subplot(4, 1, 3)
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.legend(loc = 'upper right')
plt.title('MSE')

plt.subplot(4, 1, 4)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.legend(loc = 'upper right')
plt.title('MAE')

plt.show()


# In[43]:


# plotting the confusion matrix 

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    
        print(cm)
    
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
cnf_matrix = confusion_matrix(np.argmax(Y_test, axis=1), y_pred)

plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=mods, normalize=True,
                        title='Normalized confusion matrix')
plt.show()

