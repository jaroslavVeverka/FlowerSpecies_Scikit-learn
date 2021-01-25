import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

train_path = "dataset/train"
images_per_class = 80
fixed_size = tuple((300, 300))
h5_data = 'output/dataCNN.h5'
h5_labels = 'output/labelsCNN.h5'

train_labels = os.listdir(train_path)
print(train_labels)

images = []
labels = []

for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)
    print(dir)

    # get the current training label
    current_label = training_name

    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)
        image = image.astype('float32')
        image = image / 255.
        images.append(image)

        labels.append(current_label)


#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaled_features = scaler.fit_transform(images)

print(np.array(images).shape)
print(np.array(labels).shape)

targetNames = os.listdir('dataset/train/')
print(targetNames)
targets = np.array([targetNames.index(label) for label in labels])
print(targets)

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(images))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(targets))

h5f_data.close()
h5f_label.close()

from sklearn.model_selection import train_test_split
train_X, test_X, train_label, test_label = train_test_split(np.array(images), targets, test_size=0.2, random_state=13)
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_label, test_size=0.2, random_state=13)

print("Train labels  : {}".format(train_label.shape))
print("Train data  : {}".format(train_X.shape))

print("Valid labels  : {}".format(valid_label.shape))
print("Valid data : {}".format(valid_X.shape))

print("Test labels  : {}".format(test_label.shape))
print("Test data : {}".format(test_X.shape))

print(test_label.shape)
print(test_X.shape)

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

batch_size = 64
epochs = 10
num_classes = 10

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

model_train_test = model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(valid_X, valid_label))

test_eval = model.evaluate(test_X, test_label, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])