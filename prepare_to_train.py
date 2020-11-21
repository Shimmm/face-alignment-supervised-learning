# -*- coding: utf-8 -*-

import os
import numpy as np

import cv2

path_train = "mydataset/train/"
path_test = "mydataset/test/"


n_pts = 68
landmarks_set = []
image_set = []
for root, dirs, files in os.walk(path_train, topdown=False):
    x0 = []
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(path_train + file)
            descripteur = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_set.append(gray)
            pt_train = np.load(path_train+file[:-4]+".npy")

            landmarks_set.append(pt_train)

                
image_set = np.array(image_set)      
landmarks_set = np.array(landmarks_set)         


print("landmarks set est de dimension ", landmarks_set.shape)
np.save("mydataset/landmarks_train_set.npy", landmarks_set)
print("image set est de dimension ", image_set.shape)
np.save("mydataset/image_train_set.npy", image_set)


n_pts = 68
landmarks_set = []
image_set = []

for root, dirs, files in os.walk(path_test, topdown=False):
    x0 = []
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(path_test + file)
            descripteur = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_set.append(gray)
            pt_test = np.load(path_test+file[:-4]+".npy")

            landmarks_set.append(pt_test)

                
image_set = np.array(image_set)      
landmarks_set = np.array(landmarks_set)         


print("landmarks set est de dimension ", landmarks_set.shape)
np.save("mydataset/landmarks_test_set.npy", landmarks_set)
print("image set est de dimension ", image_set.shape)
np.save("mydataset/image_test_set.npy", image_set)
