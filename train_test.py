# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:59:59 2020

@author: HASEE
"""

import numpy as np
from regresseur import Cascade, descriptor
import matplotlib.pyplot as plt
import cv2
import time



path_train = "mydataset/train/"
path_test = "mydataset/test/"

def drawKeypoints(img, moy, kp):
    plt.figure()
    plt.imshow(img)
    # plt.plot(ver[:,0], ver[:,1], 'g.')
    plt.plot(moy[:,0], moy[:,1], 'b.')
    plt.plot(kp[:,0], kp[:,1], 'r.')
    
def train_cascade():

    start_time = time.time()

    image_set = np.load("mydataset/image_train_set.npy")
    landmarks_set = np.load("mydataset/landmarks_train_set.npy")

    n_samp, n_pts, dim = landmarks_set.shape
    landmarks_set = landmarks_set.reshape((n_samp, n_pts*dim))

    model_moyen = np.load(path_train+"train_pts_moyens.npy")
    model_moyen = np.repeat(model_moyen[np.newaxis, :, :], image_set.shape[0], axis=0)

    landmarks_set_aug = landmarks_set
    for i in range(1,11):
        perturb = np.load(path_train+str(i)+"_perturbation.npy")
        perturb = np.repeat(perturb[np.newaxis, :, :], image_set.shape[0], axis=0)
        model_moyen = np.concatenate((model_moyen, perturb),axis=0)
        landmarks_set_aug = np.concatenate((landmarks_set_aug, landmarks_set),axis=0)

    print(landmarks_set_aug.shape, model_moyen.shape, image_set.shape)
    print("-------> dataset augmented preprared!")

    n_iter = 5
    myCascade = Cascade()
    myCascade.fit(image_set, model_moyen, landmarks_set_aug)
    myCascade.save_model()

    print("time taken:", (time.time() - start_time)/60)

def test_cascade():

    image_set = np.load("mydataset/image_test_set.npy")
    landmarks_set = np.load("mydataset/landmarks_test_set.npy")
    n_samp, n_pts, dim = landmarks_set.shape
    # landmarks_set = landmarks_set.reshape((n_samp, n_pts*dim))

    pt_moyen = np.load(path_train+"train_pts_moyens.npy")
    pts_moyen = np.repeat(pt_moyen[np.newaxis, :, :], image_set.shape[0], axis=0)



    R = np.load("mydataset/R3.npy")
    A = np.load("mydataset/A3.npy")
    sift = cv2.xfeatures2d.SIFT_create()
    n_samp, n_pts, dim = pts_moyen.shape

    for i in range(len(R)):
        
        R1 = R[i]
        A1 = A[i]
        
        X0_test = descriptor(pts_moyen, image_set, sift)
        X0_test1 = X0_test.dot(A1.T)
        Y1 = np.concatenate((np.ones((X0_test1.shape[0],1)), X0_test1), axis=1)
        Qs_test1 = Y1.dot(R1)
        # print(Qs_test1.shape, Qs_star.shape)
        keypoints = pts_moyen.reshape((n_samp, n_pts*dim))
        keypoints = keypoints + Qs_test1
        pts_moyen = keypoints.reshape((n_samp, n_pts, dim))
        
        keypoints = keypoints.reshape((pts_moyen.shape))
        drawKeypoints(image_set[1], landmarks_set[1], keypoints[1])
        drawKeypoints(image_set[0], landmarks_set[0], keypoints[0])
        # drawKeypoints(image_set[17], pt_moyen, keypoints[17])
        # drawKeypoints(image_set[16], pt_moyen, keypoints[16])
        print("========> Iteration ", str(i+1))

# train_cascade()
test_cascade()



    
