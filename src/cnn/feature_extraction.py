import os
import cv2

def prepare_images(data_dir, img_size=300):
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
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                resized_arr = resized_arr / 255.0
                data.append([class_num, resized_arr])
            except Exception as e:
                print(e)
        print('[STATUS] images from', path, 'prepared')
        print('[STATUS] ', class_num)
    return data

