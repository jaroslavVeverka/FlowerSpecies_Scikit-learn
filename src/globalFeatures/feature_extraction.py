import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mahotas
import h5py

bins = 8


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


def extract_global_features(images, img_size=300):
    labeled_featured_images = []
    print('[STATUS] extracting global featured from', len(images), 'images')
    for i, image in enumerate(images):
        resized_arr = cv2.resize(image, (img_size, img_size))

        fv_hu_moments = fd_hu_moments(resized_arr)
        fv_haralick = fd_haralick(resized_arr)
        fv_histogram = fd_histogram(resized_arr)

        labeled_featured_images.append(np.hstack([fv_hu_moments, fv_haralick, fv_histogram]))

        if (i + 1) % 100 == 0:
            print('[STATUS]', i + 1, 'images processed')

    print('[STATUS] feature extraction of', i + 1, 'images processed')
    return labeled_featured_images

# data_dir = 'C:/Users/jarda/IdeaProjects/FlowerSpecies_Scikit-learn/src/dataset/train/'

# # get and sort labels
# labels = os.listdir(data_dir)
# labels.sort()
# print(labels)

# labeled_images = prepare_images(data_dir)

# show prepared image
# cv2.imshow('image',images[-1][1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# images = [image[1] for image in labeled_images]
# labels = [image[0] for image in labeled_images]
#
# print(f'[STATUS] data size: ', np.array(labeled_images).shape)
# extracted_features = extract_global_features(images)
#
# labeled_extracted_features = pd.concat([pd.DataFrame(labels), pd.DataFrame(extracted_features)], axis=1)
#
# print(f'[STATUS] feature vector size ', labeled_extracted_features.shape)
#
# h5_data_path = 'C:/Users/jarda/IdeaProjects/FlowerSpecies_Scikit-learn/src/globalFeatures/h5_data'
#
# h5f_data = h5py.File(h5_data_path, 'w')
# h5f_data.create_dataset('dataset_1', data=labeled_extracted_features)
#
# h5f_data.close()
