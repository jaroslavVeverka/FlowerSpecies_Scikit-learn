############################
# GLOBAL FEATURE EXTRACTION
############################
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

#####################
# tunable-parameters
#####################

images_per_class = 80
fixed_size = tuple((500, 500))
train_path = "dataset/train"
h5_data = 'output/data.h5'
h5_labels ='output/labels.h5'
bins = 8


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
