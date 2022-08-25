#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 12:23:05 2022

@author: filippo
"""

"""
Python script aimed at importing libraries to be used in the Google Colab Jupyter notebooks
"""

import os
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Import data for deep learning')

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

#### Set up of parameters and libraries
## SETTINGS #######################

####################################

## FUNCTIONS
def set_seeds(n):

  #general random seed
  seed(n)
  #tensorflow-specific seed
  tf.random.set_seed(n)

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

    if K.image_data_format() == 'channels_first': #from keras backend as K
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255 #max value of pixel intensity
    X_test /= 255 #max value of pixel intensity

    # convert class vectors to binary class matrices (also known as OHE - One Hot Encoding)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return(X_train,X_test,y_train,y_test)


print("DONE!")
