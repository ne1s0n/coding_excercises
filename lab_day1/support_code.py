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
  
## numpy
import numpy as np
from numpy.random import seed

#### Set up of parameters and libraries
## SETTINGS #######################

# where the kinship data are stored
#remote_data_folder = 'http://www.jackdellequerce.com/data/cattle/kinships_sorted/'
#remote_data_folder = os.path.join(args.remote_folder,'')

# where to place the data
#downloaded_data = '/content/data/'
#downloaded_data = args.target_dir
#downloaded_data = '/home/filippo/Documents/deep_learning_for_breeding/'

# specific dataset
#dataset='cattle/'
#dataset = args.dataset

#base_dir = downloaded_data + dataset
#base_dir = os.path.join(downloaded_data, dataset, '')
####################################

print("DONE!")
