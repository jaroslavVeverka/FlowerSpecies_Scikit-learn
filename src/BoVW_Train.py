from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import os
import h5py
import re
import pylab as pl
from sklearn.metrics import confusion_matrix, accuracy_score

images_per_class = 80
fixed_size = tuple((500, 500))
train_path = "dataset/train"
h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'
bins = 8

train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

features_descriptors = []
labels = []

brisk = cv2.BRISK_create(30)

import operator
#oper = operator.itemgetter(6, 13)
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        kpts, des = brisk.detectAndCompute(image, None)

        # update the list of labels and feature vectors
        labels.append(current_label)
        features_descriptors.append(des)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(features_descriptors).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

#encode the target labels
# targetNames = np.unique(labels)
# le = LabelEncoder()
# target = le.fit_transform(labels)
# print(target)
# print("[STATUS] training labels encoded...")


targetNames = os.listdir('dataset/train/')
print(targetNames)
target = np.array([targetNames.index(lable) for lable in labels])
print(target)
print("[STATUS] training labels encoded...")

descriptors = features_descriptors[0][1]
for des in features_descriptors:
    descriptors = np.vstack((descriptors, des))
  
print(descriptors.shape)

descriptors_float = descriptors.astype(float)

from scipy.cluster.vq import kmeans, vq
k = 200
voc, variance = kmeans(descriptors_float, k, 1)

# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
im_features = np.zeros((len(target), k), "float32")
for i in range(len(target)):
    words, distance = vq(features_descriptors[i],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(target)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

print(im_features.shape)

from sklearn.model_selection import train_test_split
# split the training and testing data
trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal = train_test_split(np.array(im_features),
                                                                                        np.array(target),
                                                                                        test_size=0.2,
                                                                                        random_state=9)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=10000)
clf.fit(trainDataGlobal, trainLabelsGlobal)

prediction = clf.predict(testDataGlobal)

accuracy = accuracy_score(testLabelsGlobal, prediction)
print("accuracy = ", accuracy)
cm = confusion_matrix(testLabelsGlobal, prediction)
print(cm)

import joblib
joblib.dump((clf, train_labels, stdSlr, k, voc), "bovw.pkl", compress=3)