
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


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()


global_features_test = []
labels_test = []

print(os.listdir(test_path))

labels_test = [re.search(regex, picture_name).group(0) for picture_name in os.listdir(test_path)]

print(labels_test)

for file in os.listdir(test_path):
    image = cv2.imread(test_path + file)
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick = fd_haralick(image)
    fv_histogram = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    global_features_test.append(global_feature)


print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features_test).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels_test).shape))

# encode the target labels
targetNames = os.listdir('dataset/train/')
print(targetNames)
target = np.array([targetNames.index(lable) for lable in labels_test])
print(target)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
#scaler = MinMaxScaler(feature_range=(0, 1))
#rescaled_features = scaler.fit_transform(global_features_test)
rescaled_features = global_features_test
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data_test, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels_test, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] completed Global Feature Extraction...")

