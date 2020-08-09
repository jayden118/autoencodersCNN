#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 23:49:02 2020

@author: Weihong Chin
"""
import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import sys

import keras
import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate
from keras import regularizers
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import cifar10

from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


"""
Dataset Preparation - cifar10 
"""
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Create a dictionary for visualization later
dict_cifar = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# Generate imbalanced dataset
x_test_extract = []
y_test_extract = []
x_train_imbalance = []
y_train_imbalance = []
count_class = [0, 0, 0]

for i, j in zip(X_train, Y_train):
    # extract first 2000 bird, deer, truck images from X_train dataset as test data
    if (j == 2): # bird class
        if (count_class[0] < 2000): 
            x_test_extract.append(i)
            y_test_extract.append(j)
            count_class[0] += 1
        else:
            x_train_imbalance.append(i)
            y_train_imbalance.append(j)
            
    elif (j == 4): # deer class
        if (count_class[1] < 2000): 
            x_test_extract.append(i)
            y_test_extract.append(j)
            count_class[1] += 1
        else:
            x_train_imbalance.append(i)
            y_train_imbalance.append(j)
            
    elif (j == 9): # truck class
        if (count_class[2] < 2000):
            x_test_extract.append(i)
            y_test_extract.append(j)
            count_class[2] += 1
        else:
            x_train_imbalance.append(i)
            y_train_imbalance.append(j)
            
    else:
        x_train_imbalance.append(i)
        y_train_imbalance.append(j)
        

# Convert the extracted dataset as numpy array
x_test_extract = np.array(x_test_extract)
y_test_extract = np.array(y_test_extract)      
x_train_imbalance = np.array(x_train_imbalance)
y_train_imbalance = np.array(y_train_imbalance)     

# Append the original test set with extracted test set 
x_test_imbalance = np.append(X_test, x_test_extract, axis=0)
y_test_imbalance = np.append(Y_test, y_test_extract, axis=0)

# Data Normalization
x_train_imbalance = x_train_imbalance.astype('float32') # convert unit8 to float32 - prevent overflow
x_test_imbalance = x_test_imbalance.astype('float32')  
x_train_imbalance = x_train_imbalance / 255.
x_test_imbalance = x_test_imbalance / 255.

# Randomly split 20% of training data as validation data
x_train, x_valid, y_trainT, y_validT = train_test_split(x_train_imbalance, y_train_imbalance, 
                                                        test_size=0.2, random_state=42, shuffle=True)

# Convert Target to One-Hot Encoding - avoid numerical number order relationship
y_train = np_utils.to_categorical(y_trainT, 10)
y_valid = np_utils.to_categorical(y_validT, 10)
y_test_imbalance_oneHot = np_utils.to_categorical(y_test_imbalance)

BATCH_SIZE = 32 # 32, 64, 128, 256
EPOCHS = 100

class_weights = class_weight.compute_class_weight('balanced', np.unique(y_trainT), 
                                                  y_trainT.reshape(y_trainT.shape[0]))
class_weights

sample_weights = np.ones(shape=(len(y_trainT),))
sample_weights[y_trainT[:,0] == 2] = 2.0
sample_weights[y_trainT[:,0] == 4] = 2.0
sample_weights[y_trainT[:,0] == 6] = 2.0

"""
Implement UNET
"""
def create_block(inputs, channels): # 2 layers of Convolution block
    x = inputs
    for i in range(2):
        x = Conv2D(channels, 3, padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
    return x


def conv_AE():
    inputs = Input((32, 32, 3))

    # Encoder
    x = Conv2D(64, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), padding='same')(x)
    x = Conv2D(32, kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2), padding='same')(x)
    x = Conv2D(16, kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Bottleneck - latent space
    encoded = MaxPool2D(pool_size=(2,2), padding='same')(x)

    # Decoder
    x = Conv2D(16, kernel_size=(3,3), padding='same')(encoded)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(32, kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(3, kernel_size=(3,3), padding='same')(x)
    x = BatchNormalization()(x)

    # Output
    decoded = Activation('sigmoid')(x)

    return Model(inputs, encoded), Model(inputs, decoded)


def uNet():
    inputs = Input((32, 32, 3))
    
    # Encoder block
    block1 = create_block(inputs, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    
    # Bottleneck - latent space
    encoded = create_block(x, 128)
    
    # Decoder block
    x = Conv2DTranspose(64, kernel_size=2, strides=2)(x)
    x = Concatenate()([block2, x])
    x = create_block(x, 64)
    x = Conv2DTranspose(32, kernel_size=2, strides=2)(x)
    x = Concatenate()([block1, x])
    x = create_block(x, 32)
    
    # output layer
    x = Conv2D(3, 1)(x)
    decoded = Activation("sigmoid")(x)
    
    return Model(inputs, encoded), Model(inputs, decoded)

# function for selecting general AE or UNET
def train_AE(model_name):
    if model_name == 'unet':
        encoder, model = uNet()
        
    elif model_name == 'cae':
        encoder, model = conv_AE()
    else:
        print("Wrong model name")
        
    return encoder, model


# implement UNET 
encoder_unet, model_unet = train_AE('unet')
model_unet.compile(optimizer='adam', loss='mean_squared_error')
model_unet.summary() 
    
es = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
#lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=5, min_delta=0.0001)
CALLBACKS = [es]
history = model_unet.fit(x_train, x_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS, verbose=1,
                         validation_data=(x_valid, x_valid),
                         shuffle=True, callbacks=CALLBACKS,
                         sample_weight=sample_weights)

gist_train_unet = encoder_unet.predict(x_train)
gist_valid_unet = encoder_unet.predict(x_valid)
gist_test_unet = encoder_unet.predict(x_test_imbalance)

"""
Optimization
"""
def f_nn(params):   
    print ('Params testing: ', params)
    model = Sequential()
    model.add(Conv2D(params['units1'], 3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D())
    model.add(Conv2D(params['units2'], 3, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPool2D())   

    model.add(Flatten())
    model.add(Dense(output_dim=params['units4'], activation="relu"))
    model.add(Dropout(params['dropout1']))
    model.add(Dense(output_dim=params['units5'], activation="relu"))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

    model.fit(gist_train_unet, y_train, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    acc = model.evaluate(gist_valid_unet, y_valid)[1]
    print('Accuracy:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


space = {
            'units1': hp.choice('units1', [256,512,1024]),
            'units2': hp.choice('units2', [128,256,512]),
            'units4': hp.choice('units4', [256,512,1024]),
            'units5': hp.choice('units5', [50,64,100,128]),
            'dropout1': hp.uniform('dropout1', .25,.5),
            'dropout2': hp.uniform('dropout2', .25,.5),
            'batch_size' : hp.choice('batch_size', [32,64,128,256,512]),
         
            'nb_epochs' :  50,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=5, trials=trials)

print('best: ')
print(best)

