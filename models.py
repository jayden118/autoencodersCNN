#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 13:52:01 2020

@author: Weihong Chin
"""
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import sys
import random as python_random
from subprocess import call

import keras
import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, MaxPool2D, Flatten, BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation, Add, Concatenate
from keras import regularizers
from keras.models import Model, Sequential, save_model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.objectives import mean_squared_error
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import cifar10

from sklearn.metrics import classification_report, confusion_matrix

from numpy.random import seed
seed(7)
python_random.seed(7)
from tensorflow import set_random_seed
set_random_seed(8)


"""
Helper Function
"""
# Generate classification report and confusion matrix
def generate_report(predictions, y_test_imbalance_oneHot, dict_cifar):
    cm = confusion_matrix(y_test_imbalance_oneHot.argmax(axis=1), predictions.argmax(axis=1))

    print("Classification report: \n")
    cr = classification_report(y_test_imbalance_oneHot.argmax(axis=1),
                               predictions.argmax(axis=1),
                               target_names=list(dict_cifar.values()))
    print(cr)

    ax = sb.heatmap(cm, annot=True,
               xticklabels=list(dict_cifar.values()), yticklabels=list(dict_cifar.values()),
               cmap="YlGnBu", fmt="d")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.show()


# Loss function for autoencoder
def loss_function(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred)
    return K.sum(MSE, axis=(1, 2))


# Create 2 layers of Convolution block
def create_block(inputs, channels):
    x = inputs
    for i in range(2):
        x = Conv2D(channels, kernel_size=(3,3), padding='same')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
    return x


"""
CNN for benchmarking
"""
def cnn():
    inputs = Input((32, 32, 3))
    block1 = create_block(inputs, 32)
    x = MaxPool2D(2)(block1)
    block2 = create_block(x, 64)
    x = MaxPool2D(2)(block2)
    block3 = create_block(x, 128)
    x = Dropout(0.4)(block3)
    x = Flatten()(x)
    output = Dense(10, activation='softmax')(x)
    return Model(inputs, output)


def train_cnn(x_train, y_train, batch_size, epochs, x_valid, y_valid, class_weights, data_aug):
    classifier = cnn()
    classifier.summary()
    classifier.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    CALLBACKS = [es]

    if not data_aug:
        history = classifier.fit(x_train, y_train,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1, callbacks=CALLBACKS,
                                 validation_data=(x_valid, y_valid),
                                 class_weight=class_weights)
    else:
        train_datagen = ImageDataGenerator(shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        training_set = train_datagen.flow(x_train, y_train,
                                          batch_size=batch_size)

        validation_datagen = ImageDataGenerator()

        validation_set = validation_datagen.flow(x_valid, y_valid,
                                                 batch_size=batch_size)

        history = classifier.fit_generator(training_set,
                                           epochs=epochs,
                                           steps_per_epoch=np.ceil(x_train.shape[0]/batch_size),
                                           verbose=1, callbacks=CALLBACKS,
                                           validation_data=(validation_set),
                                           validation_steps=np.ceil(x_valid.shape[0]/batch_size),
                                           class_weight=class_weights)
    return history, classifier


"""
Autoencoder Model
"""
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


def simplified_uNet():
    inputs = Input((32, 32, 3))

    # Encoder block
    conv1 = create_block(inputs, 32)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = create_block(pool1, 64)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)

    # Bottleneck - latent space
    encoded = create_block(pool2, 128)

    # Decoder block
    trans1 = Conv2DTranspose(64, kernel_size=(2,2), strides=2)(encoded)
    up1 = Concatenate()([conv2, trans1])
    conv3 = create_block(up1, 64)
    trans2 = Conv2DTranspose(32, kernel_size=(2,2), strides=2)(conv3)
    up2 = Concatenate()([conv1, trans2])
    conv4 = create_block(up2, 32)

    # output layer
    decoded = Conv2D(3, 1)(conv4)
    decoded = Activation("sigmoid")(decoded)

    return Model(inputs, encoded), Model(inputs, decoded)


# function for selecting general AE or UNET
def train_AE(model_name, x_train, y_train, batch_size, epochs, x_valid, y_valid, sample_weights):
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)
    CALLBACKS = [es]

    if model_name == 'sunet':
        encoder, model = simplified_uNet()

    elif model_name == 'cae':
        encoder, model = conv_AE()

    else:
        print("Wrong model name")

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') # mean_squared_error
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs, verbose=1,
                        validation_data=(x_valid, y_valid),
                        shuffle=True, callbacks=CALLBACKS,
                        sample_weight=sample_weights)
    return history, encoder, model


# Train CNN with autoencoder feature
def classifier_conv(inputs):
    inputs = Input((inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    x = Conv2D(1024, 3, padding='same')(inputs)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    return Model(inputs, output)


def train_classifier_conv(x_train, y_train, batch_size, epochs, x_valid, y_valid, class_weights):
    classifier = classifier_conv(x_train)
    classifier.summary()
    classifier.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_acc', patience=10, restore_best_weights=True)
    CALLBACKS = [es]

    history = classifier.fit(x_train, y_train,
                             batch_size=batch_size,
                             epochs=epochs,
                             verbose=1, callbacks=CALLBACKS,
                             validation_data=(x_valid, y_valid),
                             class_weight=class_weights)

    return history, classifier


# End to End Simplified UNET Autoencoder with CNN
def ae_cnn_end2end():
    inputs = Input((32,32,3))

    # Encoder
    conv1 = create_block(inputs, 32)
    pool1 = MaxPool2D(pool_size=(2,2))(conv1)
    conv2 = create_block(pool1, 64)
    pool2 = MaxPool2D(pool_size=(2,2))(conv2)

    # Bottleneck - latent space
    encoded = create_block(pool2, 128)

    # Decoder
    trans1 = Conv2DTranspose(64, kernel_size=(2,2), strides=2)(encoded)
    up1 = Concatenate()([conv2, trans1])
    conv3 = create_block(up1, 64)
    trans2 = Conv2DTranspose(32, kernel_size=(2,2), strides=2)(conv3)
    up2 = Concatenate()([conv1, trans2])
    conv4 = create_block(up2, 32)

    # reconstruction
    conv5 = Conv2D(3, 1)(conv4)
    decoded = Activation("sigmoid", name='autoencoder')(conv5)

    #classification
    c = Conv2D(1024, 3, padding="same")(encoded)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(128, 3, padding="same")(c)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.4)(c)
    c = Flatten()(c)
    c = Dense(512, activation='relu')(c)
    c = Dropout(0.4)(c)
    c = Dense(100, activation='relu')(c)
    c = Dropout(0.5)(c)
    classify = Dense(10, activation='softmax', name='classification')(c)

    outputs = [decoded, classify]

    return Model(inputs, outputs)


def train_AECNN_end2end(x_train, y_train, batch_size, epochs, x_valid, y_valid, class_weights):
    multimodel = ae_cnn_end2end()
    multimodel.compile(optimizer= Adam(learning_rate=0.001), #Adam(learning_rate=0.001), #SGD(lr= 0.01, momentum=0.9),
                       loss = {'classification':'categorical_crossentropy', 'autoencoder':'mean_squared_error'},
                       loss_weights = {'classification':0.9, 'autoencoder':0.1},
                       metrics = {'classification':['accuracy'], 'autoencoder':[]})

    es = EarlyStopping(monitor='val_classification_acc', patience=10, restore_best_weights=True)
    CALLBACKS = [es]

    history_e2e = multimodel.fit(x_train, [x_train,y_train],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_valid, [x_valid,y_valid]),
                              shuffle=True, callbacks=CALLBACKS,
                              class_weight={'classification':class_weights})

    return history_e2e, multimodel


# End to End Denoising Autoencoder with CNN
def dae_cnn_end2end():
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
    decoded = Activation('sigmoid', name='autoencoder')(x)

    #classification
    c = Conv2D(1024, 3, padding="same")(encoded)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(128, 3, padding="same")(c)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.4)(c)
    c = Flatten()(c)
    c = Dense(512, activation='relu')(c)
    c = Dropout(0.4)(c)
    c = Dense(100, activation='relu')(c)
    c = Dropout(0.5)(c)
    classify = Dense(10, activation='softmax', name='classification')(c)

    outputs = [decoded, classify]

    return Model(inputs, outputs)


def train_DAECNN_end2end(x_train, y_train, batch_size, epochs, x_valid, y_valid, x_train_noise, x_valid_noise, class_weights):
    multimodel = dae_cnn_end2end()
    multimodel.compile(optimizer=Adam(learning_rate=0.001),
                       loss = {'classification': 'categorical_crossentropy', 'autoencoder': 'mean_squared_error'},
                       loss_weights = {'classification': 0.9, 'autoencoder': 0.1},
                       metrics = {'classification': ['accuracy'], 'autoencoder': []})

    es = EarlyStopping(monitor='val_classification_acc', patience=10, restore_best_weights=True)
    CALLBACKS = [es]

    history_e2e = multimodel.fit(x_train_noise, [x_train,y_train],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_valid_noise, [x_valid,y_valid]),
                              shuffle=True, callbacks=CALLBACKS,
                              class_weight={'classification':class_weights})

    return history_e2e, multimodel


# End to End Convolutional Autoencoder with CNN
def cae_cnn_end2end():
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
    decoded = Activation('sigmoid', name='autoencoder')(x)

    #classification
    c = Conv2D(1024, 3, padding="same")(encoded)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.3)(c)
    c = Conv2D(256, 3, padding="same")(c)
    c = Activation('relu')(c)
    c = BatchNormalization()(c)
    c = MaxPool2D(2)(c)
    c = Dropout(0.3)(c)
    c = Flatten()(c)
    c = Dense(256, activation='relu')(c)
    c = Dropout(0.3)(c)
    c = Dense(128, activation='relu')(c)
    c = Dropout(0.3)(c)
    classify = Dense(10, activation='softmax', name='classification')(c)

    outputs = [decoded, classify]

    return Model(inputs, outputs)


def train_CAECNN_end2end(x_train, y_train, batch_size, epochs, x_valid, y_valid, class_weights):
    multimodel = cae_cnn_end2end()
    multimodel.compile(optimizer=Adam(learning_rate=0.001),
                       loss = {'classification':'categorical_crossentropy', 'autoencoder':'mean_squared_error'},
                       loss_weights = {'classification':0.8, 'autoencoder':0.2},
                       metrics = {'classification':['accuracy'], 'autoencoder':[]})

    es = EarlyStopping(monitor='val_classification_acc', patience=5, restore_best_weights=True)
    CALLBACKS = [es]

    history_e2e = multimodel.fit(x_train, [x_train,y_train],
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(x_valid, [x_valid,y_valid]),
                              shuffle=True, callbacks=CALLBACKS,
                              class_weight={'classification':class_weights})

    return history_e2e, multimodel