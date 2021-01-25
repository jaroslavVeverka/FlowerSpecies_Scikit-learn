# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:57:43 2021

@author: jveverka
"""

import numpy as np
from sklearn.model_selection import train_test_split

# import custom functions for global features extraction
from src.cnn.feature_extraction import prepare_images

# path to data
data_dir = '../dataset/train/'

def prepare_split_images():

    # get labeled images
    labeled_images = prepare_images(data_dir)
    print('[STATUS] data size: ', np.array(labeled_images).shape)
    
    images = [image[1] for image in labeled_images]
    # get Y
    labels = [image[0] for image in labeled_images]
    
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(images),
                                                        np.array(labels),
                                                        test_size=0.2, stratify=np.array(labels), random_state=42)
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                          y_train,
                                                          test_size=0.2, stratify=np.array(y_train), random_state=42)
    
    return  X_train, X_test, X_valid, y_train, y_test, y_valid
