
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import re

#####################
# tunable-parameters
#####################
fixed_size = tuple((500, 500))
test_path = "test/"
h5_data_test = 'output/data_test.h5'
h5_labels_test = 'output/labels_test.h5'
bins = 8
regex = r"\D*[^\d*.jpg]"

import joblib
# Load the classifier, class names, scaler, number of clusters and vocabulary
#from stored pickle file (generated during training)
clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")

train_labels = os.listdir(test_path)

# sort the training labels
train_labels.sort()
print(train_labels)

features_descriptors = []
labels = []

labels = [re.search(regex, picture_name).group(0) for picture_name in os.listdir(test_path)]

print(labels)

brisk = cv2.BRISK_create(30)

for file in os.listdir(test_path):
        # read the image and resize it to a fixed-size
        image = cv2.imread(test_path + file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        kpts, des = brisk.detectAndCompute(image, None)

        # update the list of labels and feature vectors
        features_descriptors.append(des)


print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(features_descriptors).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = os.listdir('dataset/train/')
print(targetNames)
target = np.array([targetNames.index(lable) for lable in labels])
print(target)
print("[STATUS] training labels encoded...")

# Stack all the descriptors vertically in a numpy array
descriptors = features_descriptors[0][1]
for des in features_descriptors:
    descriptors = np.vstack((descriptors, des))
    print(descriptors.shape)

# Calculate the histogram of features
#vq Assigns codes from a code book to observations.
from scipy.cluster.vq import vq
test_features = np.zeros((len(target), k), "float32")
for i in range(len(target)):
    words, distance = vq(features_descriptors[i], voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(target)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
#Standardize features by removing the mean and scaling to unit variance
#Scaler (stdSlr comes from the pickled file we imported)
test_features = stdSlr.transform(test_features)
print(test_features.shape)

prediction = clf.predict(test_features)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(target, prediction)
print("accuracy = ", accuracy)
cm = confusion_matrix(target, prediction)
print(cm)

