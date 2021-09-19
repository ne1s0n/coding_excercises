###################################################
## script to train a DL model for image recognition
###################################################

"""
force keras to use CPU instead of GPU
(comment out if you don't want this)
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

"""
import libraries
"""
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K ##
from keras.datasets import mnist
from keras.models import model_from_json
import numpy

"""
Configuration parameters
"""
batch_size = 128 ## n. of samples/records in each batch
num_classes = 10
epochs = 12

# input image dimensions (pixels)
img_rows, img_cols = 28, 28

"""
Input data
"""
# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

## subsampling the data
X_train = X_train[0:10000,]
y_train = y_train[0:10000]
X_test = X_test[0:1000,]
y_test = y_test[0:1000]

print("Size of the training set")
print(X_train.shape)
print("Size of the test set")
print(X_test.shape)

"""
Data preprocessing
"""
K.image_data_format()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("Modified array dimensions:")
print(X_train.shape)  

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train[0])

"""
Model building
"""
model = Sequential()
model.add(
          Conv2D(32, kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape))


print(model.output_shape)

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

"""
Compiling the model
"""
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

"""
Training the model
"""
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

"""
Save out the trained model
"""
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")

print("Saved model to disk")

