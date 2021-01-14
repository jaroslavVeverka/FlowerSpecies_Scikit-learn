# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:20:28 2021

@author: jveverka
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mahotas
import h5py
from scipy.cluster.vq import kmeans, vq

def prepare_images(data_dir):
    labels = os.listdir(data_dir)
    labels.sort()
    print(labels)
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                data.append([class_num, img_arr])
            except Exception as e:
                print(e)
        print(f'[STATUS] images from', path, 'prepared')
        print(f'[STATUS] ', class_num)
    return data



def extract_local_features(images, img_size=300):
    brisk = cv2.BRISK_create(30)
    #sift = cv2.SIFT_create()
    #surf = cv2.SURF(400)
    labeled_featured_images = []
    print('[STATUS] extracting local featured from', len(images), 'images')
    for i, image in enumerate(images):
        resized_arr = cv2.resize(image, (img_size, img_size))
        
        kpts, des = brisk.detectAndCompute(resized_arr, None)
        
        # create picture with detected kpts
        if (i == 0):
            print(len(kpts))
            print(brisk.descriptorSize())
            img = cv2.drawKeypoints(resized_arr, kpts, resized_arr,
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('brisk_keypoints.jpg',img)


        labeled_featured_images.append(des)

        if (i + 1) % 100 == 0:
            print('[STATUS]', i + 1, 'images processed')

    print('[STATUS] feature extraction of', i + 1, 'images processed')
    return labeled_featured_images


def fit_transform_bovw(data, k = 200):
    # split all arrays into one
    descriptors = np.vstack(data)
    descriptors = descriptors.astype(float)
    
    voc, variance = kmeans(descriptors, k, 1)
    
    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    features = np.zeros((len(data), k), "float32")
    for i in range(len(data)):
        words, distance = vq(data[i],voc)
        for w in words:
            features[i][w] += 1
    
    return features, voc


def transform_bovw(data, fitted_kmeans, k = 200):
    # split all arrays into one
    descriptors = np.vstack(data)
    descriptors = descriptors.astype(float)
    
    voc, variance = kmeans(descriptors, k, 1)
    
    # Calculate the histogram of features and represent them as vector
    #vq Assigns codes from a code book to observations.
    features = np.zeros((len(data), k), "float32")
    for i in range(len(data)):
        words, distance = vq(data[i],fitted_kmeans)
        for w in words:
            features[i][w] += 1
        
    return features
    
    
    
    
    
    
    
