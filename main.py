#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 14:11:50 2020

@author: Weihong Chin
"""

import warnings 
warnings.filterwarnings('ignore')

import os
import models
import numpy as np
import matplotlib.pyplot as plt
from random import randint 
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
import random as python_random

from numpy.random import seed
seed(7)
python_random.seed(7)
from tensorflow import set_random_seed
set_random_seed(8)


"""
Dataset Preparation - cifar10 
"""
(X_train, Y_train), (X_test, Y_test) = models.cifar10.load_data()
print(X_train.shape)

# Create a dictionary for visualization later
dict_cifar = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

# Generate imbalanced dataset
x_test_extract = []
y_test_extract = []
x_train_imbalance = []
y_train_imbalance = []
count_class = [0, 0, 0]

# extract first 2000 bird, deer, truck images from X_train dataset as test data
for i, j in zip(X_train, Y_train):  
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

# Add noise (Salt and Pepper) to training data and validation data for Denoising AE
noise_factor = 0.1
x_train_noise = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_valid_noise = x_valid + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_valid.shape)
x_test_imbalance_noise = x_test_imbalance + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_imbalance.shape)
x_train_noise = np.clip(x_train_noise, 0., 1.)
x_valid_noise = np.clip(x_valid_noise, 0., 1.)
x_test_imbalance_noise = np.clip(x_test_imbalance_noise, 0., 1.)

# Convert Target to One-Hot Encoding - avoid numerical number order relationship for classification 
y_train = models.np_utils.to_categorical(y_trainT, 10)
y_valid = models.np_utils.to_categorical(y_validT, 10)
y_test_imbalance_oneHot = models.np_utils.to_categorical(y_test_imbalance)


# Display original images and reconstructed images of autoencoder
def display_image(original, construct, num_class):
    num_classes = num_class
    plt.figure(figsize=(20, 4))
    
    # Generating a random to get random results 
    rand_num = randint(0, 5000) 
        
    for i in range(num_classes):
        # display original images
        ax = plt.subplot(2, num_classes, i+1)
        plt.imshow(original[rand_num+i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstuction images
        ax = plt.subplot(2, num_classes, i+1+num_classes)
        plt.imshow(construct[rand_num+i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
    plt.show()


# Visualize the predicted and true labels of test data        
def display_testResult(model, test_data):
    plt.figure(figsize=(40, 8))
    
    # Generating a random to get random results 
    rand_num = randint(0, 1000) 
    
    for i in range(5):
        plt.subplot(1, 5, i+1)
        test_image = np.expand_dims(test_data[rand_num+i], axis=0)
        test_result = model.predict(test_image)
        plt.imshow(x_test_imbalance[rand_num+i])
        index = np.argsort(test_result[0, :])
        plt.title("Pred: {}, True: {}".format(dict_cifar[index[9]], 
                                              dict_cifar[y_test_imbalance[rand_num+i][0]]))
        
    plt.show()


def display_lossGraph(history, model_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name +' Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()
    
    
def display_accGraph(history, model_name):
    if model_name == 'e2e':
        plt.plot(history.history['classification_accuracy'])
        plt.plot(history.history['val_classification_accuracy'])
    else:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
    plt.title(model_name + ' Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.show()
  

"""
Training parameters
"""
BATCH_SIZE = 128 # 32, 64, 128, 256
EPOCHS = 50
saveDir = "weights/"
if not os.path.isdir(saveDir):
    os.makedirs(saveDir)

# Calculate class weights
cw = class_weight.compute_class_weight('balanced', 
                                       np.unique(y_trainT), 
                                       y_trainT.reshape(y_trainT.shape[0]))
print(cw)

class_weights ={
     0: 1.0, 
     1: 1.0, 
     2: 2.0, # round up 1.667
     3: 1.0,
     4: 2.0,
     5: 1.0,
     6: 1.0,
     7: 1.0,
     8: 1.0,
     9: 2.0
     }

# Calculate sample weights 
sw = class_weight.compute_sample_weight('balanced', 
                                        np.unique(y_trainT), 
                                        y_trainT.reshape(y_trainT.shape[0]))   
print(sw)

sample_weights = np.ones(shape=(len(y_trainT),))
sample_weights[y_trainT[:,0] == 2] = 2.0
sample_weights[y_trainT[:,0] == 4] = 2.0
sample_weights[y_trainT[:,0] == 6] = 2.0


"""
Simple CNN for benchmarking
"""
history_cnn, classifier = models.train_cnn(x_train=x_train, y_train=y_train,
                                      batch_size=BATCH_SIZE,
                                      epochs=EPOCHS,
                                      x_valid=x_valid, y_valid=y_valid,
                                      class_weights=class_weights,
                                      data_aug=1)

print('Test accuracy for benchmark CNN model: {}'.format(classifier.evaluate(x_test_imbalance, 
                                                                             y_test_imbalance_oneHot)[1]))

display_testResult(classifier, x_test_imbalance)
predictions = classifier.predict(x_test_imbalance)
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)
display_lossGraph(history=history_cnn, model_name='CNN')


"""
Implement Convolutional Auto-Encoder
"""
history_sae, encoder_sae, model_sae = models.train_AE(model_name='cae', 
                                                      x_train=x_train, y_train=x_train,
                                                      batch_size=BATCH_SIZE,
                                                      epochs=EPOCHS,
                                                      x_valid=x_valid, y_valid=x_valid,
                                                      sample_weights=sample_weights)

model_sae.save("cae.h5")
display_lossGraph(history=history_sae, model_name='Conv. AE')
recon_test_sae = model_sae.predict(x_test_imbalance)
recon_valid_sae = model_sae.predict(x_valid)
display_image(x_valid, recon_valid_sae, 10)
display_image(x_test_imbalance, recon_test_sae, 10)

# Extract SAE features
gist_train_sae = encoder_sae.predict(x_train)
gist_valid_sae = encoder_sae.predict(x_valid)
gist_test_sae = encoder_sae.predict(x_test_imbalance)

# Implement SAE_CNN
history_sae_conv, model_sae_conv = models.train_classifier_conv(x_train=gist_train_sae, y_train=y_train,
                                                                batch_size=BATCH_SIZE,
                                                                epochs=EPOCHS,
                                                                x_valid=gist_valid_sae, y_valid=y_valid,
                                                                class_weights=class_weights)

model_sae_conv.save("sae_conv.h5")
display_accGraph(history=history_sae_conv, model_name='Conv. AE_CNN')
print('Test accuracy for Conv AE_CNN model = {}'.format(model_sae_conv.evaluate(gist_test_sae, 
                                                                            y_test_imbalance_oneHot)[1]))
display_testResult(model_sae_conv, gist_test_sae)
predictions = model_sae_conv.predict(gist_test_sae)
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)


"""
Implement Denoising Convolutional Auto-Encoder
"""
history_dae, encoder_dae, model_dae = models.train_AE(model_name='cae', 
                                                      x_train=x_train_noise, y_train=x_train,
                                                      batch_size=BATCH_SIZE,
                                                      epochs=EPOCHS,
                                                      x_valid=x_valid_noise, y_valid=x_valid,
                                                      sample_weights=sample_weights)

model_dae.save("dae.h5")
display_lossGraph(history=history_dae, model_name='DAE')
recon_test_dae = model_dae.predict(x_test_imbalance_noise)
recon_valid_dae = model_dae.predict(x_valid_noise)
display_image(x_valid_noise, recon_valid_dae, 10)
display_image(x_test_imbalance_noise, recon_test_dae, 10)

# Extract DAE features
gist_train_dae = encoder_dae.predict(x_train_noise)
gist_valid_dae = encoder_dae.predict(x_valid_noise)
gist_test_dae = encoder_dae.predict(x_test_imbalance_noise)

# Implement DAE_CNN
history_dae_conv, model_dae_conv = models.train_classifier_conv(x_train=gist_train_dae, y_train=y_train,
                                                                batch_size=BATCH_SIZE,
                                                                epochs=EPOCHS,
                                                                x_valid=gist_valid_dae, y_valid=y_valid,
                                                                class_weights=class_weights)

model_dae_conv.save("dae_conv.h5")
display_accGraph(history=history_dae_conv, model_name='DAE_CNN')
print('Test accuracy for DAE_CNN model = {}'.format(model_dae_conv.evaluate(gist_test_dae, 
                                                                            y_test_imbalance_oneHot)[1]))
display_testResult(model_dae_conv, gist_test_dae)
predictions = model_dae_conv.predict(gist_test_dae)
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)


"""
Implement simplified UNET
"""
history_sunet, encoder_sunet, model_sunet = models.train_AE(model_name='sunet', 
                                                            x_train=x_train, y_train=x_train,
                                                            batch_size=BATCH_SIZE,
                                                            epochs=EPOCHS,
                                                            x_valid=x_valid, y_valid=x_valid,
                                                            sample_weights=sample_weights)

model_sunet.save("sunet.h5")
display_lossGraph(history=history_sunet, model_name='Simplified UNET')
recon_test_sunet = model_sunet.predict(x_test_imbalance)
recon_valid_sunet = model_sunet.predict(x_valid)
display_image(x_valid, recon_valid_sunet, 10)
display_image(x_test_imbalance, recon_test_sunet, 10)

# Extract SUNET features
gist_train_sunet = encoder_sunet.predict(x_train)
gist_valid_sunet = encoder_sunet.predict(x_valid)
gist_test_sunet = encoder_sunet.predict(x_test_imbalance)

# Implement Simplified UNET_CNN
history_sunet_conv, model_sunet_conv = models.train_classifier_conv(x_train=gist_train_sunet, y_train=y_train,
                                                                batch_size=BATCH_SIZE,
                                                                epochs=EPOCHS,
                                                                x_valid=gist_valid_sunet, y_valid=y_valid,
                                                                class_weights=class_weights)

model_sunet_conv.save("unet_conv.h5")
display_accGraph(history=history_sunet_conv, model_name='UNET_CNN')
print('Test accuracy for Simplified UNET_CNN model = {}'.format(model_sunet_conv.evaluate(gist_test_sunet, 
                                                                              y_test_imbalance_oneHot)[1]))
display_testResult(model_sunet_conv, gist_test_sunet)
predictions = model_sunet_conv.predict(gist_test_sunet)
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)


""" 
Implement End-to-End AutoEncoder (Simplified uNet) with CNN
"""
history_e2e, model_e2e = models.train_AECNN_end2end(x_train=x_train, y_train=y_train,
                                                         batch_size=BATCH_SIZE,
                                                         epochs=EPOCHS,
                                                         x_valid=x_valid, y_valid=y_valid,
                                                         class_weights=class_weights)

recon_valid_e2e = model_e2e.predict(x_valid)[0]
recon_test_e2e = model_e2e.predict(x_test_imbalance)[0]
display_image(x_valid, recon_valid_e2e, 10)
display_image(x_test_imbalance, recon_test_e2e, 10)

predictions = model_e2e.predict(x_test_imbalance)[1]
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)

print(history_e2e.history.keys())
display_lossGraph(history=history_e2e, model_name='e2e')
display_accGraph(history=history_e2e, model_name='e2e')


""" 
Implement End-to-End Denoising AutoEncoder with CNN
"""
history_e2e, model_e2e = models.train_DAECNN_end2end(x_train=x_train, y_train=y_train,
                                                         batch_size=BATCH_SIZE,
                                                         epochs=EPOCHS,
                                                         x_valid=x_valid, y_valid=y_valid,
                                                         x_train_noise=x_train_noise, x_valid_noise=x_valid_noise,
                                                         class_weights=class_weights)

display_lossGraph(history=history_e2e, model_name='e2e')
display_accGraph(history=history_e2e, model_name='e2e')

recon_test_e2e = model_e2e.predict(x_test_imbalance_noise)[0]
recon_valid_e2e = model_e2e.predict(x_valid_noise)[0]
display_image(x_valid_noise, recon_valid_e2e, 10)
display_image(x_test_imbalance_noise, recon_test_e2e, 10)

predictions = model_e2e.predict(x_test_imbalance_noise)[1]
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)


""" 
Implement End-to-End Convolutional AutoEncoder with CNN
"""
history_e2e, model_e2e = models.train_CAECNN_end2end(x_train=x_train, y_train=y_train,
                                                         batch_size=BATCH_SIZE,
                                                         epochs=EPOCHS,
                                                         x_valid=x_valid, y_valid=y_valid,
                                                         class_weights=class_weights)
                                                         

display_lossGraph(history=history_e2e, model_name='e2e')
display_accGraph(history=history_e2e, model_name='e2e')

recon_test_e2e = model_e2e.predict(x_test_imbalance)[0]
recon_valid_e2e = model_e2e.predict(x_valid)[0]
display_image(x_test_imbalance, recon_test_e2e, 10)
display_image(x_valid, recon_valid_e2e, 10)

predictions = model_e2e.predict(x_test_imbalance)[1]
models.generate_report(predictions, y_test_imbalance_oneHot, dict_cifar)



