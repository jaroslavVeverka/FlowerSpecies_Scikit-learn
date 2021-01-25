# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:22:58 2021

@author: jveverka
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import custom functions for global features extraction
from src.globalFeatures.feature_extraction import extract_global_features, prepare_images

# path to data
data_dir = '../dataset/train/'

# get labeled images
labeled_images = prepare_images(data_dir)
print('[STATUS] data size: ', np.array(labeled_images).shape)

images = [image[1] for image in labeled_images]
# get Y
labels = [image[0] for image in labeled_images]

# get extracted features of images: X
extracted_features = extract_global_features(images)

X_train, X_test, y_train, y_test = train_test_split(np.array(extracted_features),
                                                    np.array(labels),
                                                    test_size=0.2, stratify=np.array(labels), random_state=42)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

train_data = pd.concat([pd.DataFrame(y_train), X_train], axis=1)
test_data = pd.concat([pd.DataFrame(y_test), X_test], axis=1)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)