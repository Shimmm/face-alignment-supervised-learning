# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn.decomposition import PCA

import cv2

path_train = "mydataset/train/"
path_test = "mydataset/test/"

pt_moyen_train = np.load(path_train+"train_pts_moyens.npy")
pt_moyen_test = np.load(path_test+"test_pts_moyens.npy")
n_pts, _ = pt_moyen_train.shape


## calculate descriptor of SIFT
img = cv2.imread(path_train + str() + '.jpg')
sift = cv2.xfeatures2d.SIFT_create()

for root, dirs, files in os.walk(path_train, topdown=False):
    x0 = []
    i = 0
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(path_train + file)
            descripteur = None
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for n in range(n_pts):
                keypoints = [cv2.KeyPoint(x=pt_moyen_train[n,0],y=pt_moyen_train[n,1], _size=20)]
                kp, des = sift.compute(gray, keypoints)
                if not isinstance(descripteur, np.ndarray):
                    descripteur = des
                else:
                    descripteur = np.concatenate((descripteur, des), axis=1)
            # print(i, descripteur)
            x0.append(descripteur[0])
            i += 1
x0 = np.array(x0)               
print("x0 avant la reduction de dimensionalites est de dimension ", x0.shape)
print("x0: \n",x0)

pca = PCA(n_components=0.98)
pca.fit(x0)
x0 = pca.transform(x0)
print("x0 apres la reduction de dimensionalites est de dimension ", x0.shape)
np.save("mydataset/descripteurs_train.npy", x0)


    
