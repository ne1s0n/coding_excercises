"""
Script to evaluate the performnce
of a trained deep learning model for image recognition
on independent test data
"""

"""
force keras to use CPU instead of GPU
(comment out if you don't want this)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
Import libraries
"""
import keras
import keras.utils
import tensorflow
from keras import backend as K ## needed? What for?
from keras.datasets import mnist
from keras.models import model_from_json
import numpy as np

"""
Configuration parameters
"""
num_classes = 10
img_rows, img_cols = 28, 28

"""
Test data
"""
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## subsampling the data
X_test = X_test[0:1000,]
y_test = y_test[0:1000]

print("Size of the test set")
print(X_test.shape)

"""
Data preprocessing
"""
K.image_data_format()

if K.image_data_format() == 'channels_first':
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_test = X_test.astype('float32')
X_test /= 255
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

"""
Test model trained for image recognition
"""
# load json and create model
json_file = open('model.json', 'r')
#json_file = open('model.50k_data_100_epochs.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
#loaded_model.load_weights("model.50k_data_100_epochs.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(
	loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
	metrics=['accuracy'])

# get predictions
preds = loaded_model.predict(X_test)
res = np.apply_along_axis(func1d=np.argmax, axis=1, arr=preds)
print("predictions:", res)

print("frequency table of predicted figures:")
print(np.unique(res, return_counts=True))

# overall accuracy
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

