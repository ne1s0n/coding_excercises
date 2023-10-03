#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:23:05 2022

@author: filippo
"""

"""
Python script aimed at importing libraries to be used in the Google Colab Jupyter notebooks
and at defining all functions needed for the various steps of the building
of our first deep learning model for image recognition
"""

import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Import libraries, data and functions for deep learning')

# Add arguments
parser.add_argument('-g', '--import_all', type=str, required=False, default='yes', 
                    help='should all libraries be imported')
#parser.add_argument('-s', '--target_dir', type=str, required=True, 
#                    help='directory where data are to be stored (created if needed)')
#parser.add_argument('-d', '--dataset', type=str, required=True, 
#                    help='dataset (e.g. cattle, maize, tropical_maize, etc.)')
# Parse the argument
args = parser.parse_args()

# Print to check arguments values
print('Import all libraries:', args.import_all)

print("importing libraries")
## tensorflow and keras
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K # needed for image_data_format()
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
 
## numpy
import numpy as np
from numpy.random import seed

## matplotlib
from matplotlib import pyplot as plt

## sklearn
import sklearn.metrics

#### Set up of parameters and libraries
## SETTINGS #######################

####################################

## FUNCTIONS
print("Defining functions")
def set_seeds(n, enable_determinism=True):

  #general random seed
  #seed(n)
  #tensorflow-specific seed
  #tf.random.set_seed(n)

  # Set the seed using keras.utils.set_random_seed. This will set:
  # 1) `numpy` seed
  # 2) `tensorflow` random seed
  # 3) `python` random seed
  keras.utils.set_random_seed(n)

  # This will make TensorFlow ops as deterministic as possible, but it will
  # affect the overall performance, so it's not enabled by default.
  # `enable_op_determinism()` is introduced in TensorFlow 2.9.
  if enable_determinism:
  	tf.config.experimental.enable_op_determinism()

def load_data(ntrain,ntest):
    # the data, split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train[0:ntrain,]
    y_train = y_train[0:ntrain]
    X_test = X_test[0:ntest,]
    y_test = y_test[0:ntest]

    return (X_train,y_train,X_test,y_test)


def set_parameters(img_rows,img_cols,n_classes,batch_size,n_epochs):
    
    return (img_rows,img_cols,n_classes,batch_size,n_epochs)


def preprocess(X_train,X_test,y_train,y_test,img_rows,img_cols,num_classes):

    print("declare the correct depth of the image (channels)")
    if K.image_data_format() == 'channels_first': #from keras backend as K
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    print("normalize input data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255 #max value of pixel intensity
    X_test /= 255 #max value of pixel intensity

    # convert class vectors to binary class matrices (also known as OHE - One Hot Encoding)
    print("prepare the categorical output matrix")
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return(X_train,X_test,y_train,y_test,input_shape)


def build_model(input_shape, num_classes):

    model = Sequential()
    model.add(
          Conv2D(32, kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

def compile_model(model):

    model.compile(
            loss=tensorflow.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adadelta(),
            metrics=['accuracy'])

    print("model compiled!")

    return model


def train_model(model,X_train,y_train,batch_size,n_epochs,verbose):

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=n_epochs,
          verbose=verbose)

    return model

def evaluate_model(model,X_test,y_test):

    score = model.evaluate(X_test, y_test, verbose=0)
    #asking our model to return its predictions for the test set
    predictions = model.predict(X_test)

    #confusion_matrix function requires actual classes labels (expressed as int)
    #and not probabilities as we handled so far
    predicted_classes = predictions.argmax(axis=1)
    true_classes = y_test.argmax(axis=1)

    #rows are true values, columns are predicted values, numbering starts from zero
    confusion_matrix = sklearn.metrics.confusion_matrix(true_classes, predicted_classes)

    print("model evaluated")
    
    return (score, confusion_matrix)


##########################################################

print("DONE!")
