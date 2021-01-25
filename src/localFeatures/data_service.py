# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:33:09 2021

@author: jveverka
"""

import numpy as np
import pandas as pd

# import custom functions for global features extraction
from src.localFeatures.feature_extraction import  prepare_images, extract_local_features, fit_transform_bovw, transform_bovw
from sklearn.model_selection import train_test_split

# path to data
data_dir = '../dataset/train/'

# get labeled images
labeled_images = prepare_images(data_dir)
print('[STATUS] data size: ', np.array(labeled_images).shape)

images = [image[1] for image in labeled_images]
# get Y
labels = [image[0] for image in labeled_images]

# get extracted features of images: X
extracted_features = extract_local_features(images)

X_train, X_test, y_train, y_test = train_test_split(np.array(extracted_features),
                                                    np.array(labels),
                                                    test_size=0.2, stratify=np.array(labels), random_state=42)

print('[INFO] train X dim: ', X_train.shape)
print('[INFO] test X dim: ', X_test.shape)
print('[INFO] train Y dim: ', y_train.shape)
print('[INFO] test Y dim: ', y_test.shape)

X_train_trans_bovw, fitted_kmeans = fit_transform_bovw(X_train)
X_test_trans_bovw = transform_bovw(X_test, fitted_kmeans)

print('[INFO] X_train_trans_bovw dim: ', X_train_trans_bovw.shape)
print('[INFO] X_test_trans_bovw dim: ', X_test_trans_bovw.shape)

X_train_trans_bovw = pd.DataFrame(X_train_trans_bovw)
X_test_trans_bovw = pd.DataFrame(X_test_trans_bovw)

train_trans_bovw = pd.concat([pd.DataFrame(y_train), X_train_trans_bovw], axis=1)
test_trans_bovw = pd.concat([pd.DataFrame(y_test), X_test_trans_bovw], axis=1)

train_trans_bovw.to_csv('train_trans_bovw.csv', index=False)
test_trans_bovw.to_csv('test_trans_bovw.csv', index=False)