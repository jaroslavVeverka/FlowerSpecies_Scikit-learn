import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

from src.cnn.data_service import prepare_split_images

X_train, X_test, X_valid, y_train, y_test, y_valid = prepare_split_images()

print("Train labels  : {}".format(y_train.shape))
print("Train data  : {}".format(X_train.shape))

print("Valid labels  : {}".format(y_valid.shape))
print("Valid data : {}".format(X_valid.shape))

print("Test labels  : {}".format(y_test.shape))
print("Test data : {}".format(X_test.shape))

print(y_test.shape)
print(X_test.shape)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(17))

model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

model_train_test = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(X_valid, y_valid))

test_eval = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])