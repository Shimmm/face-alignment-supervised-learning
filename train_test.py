# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:59:59 2020

@author: SHI Mengmeng
"""

import numpy as np
from regresseur import Cascade, descriptor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cv2
import time



path_train = "mydataset/train/"
path_test = "mydataset/test/"

def drawKeypoints(img, moy, kp): # visualisation
    plt.figure()
    plt.imshow(img)
    plt.plot(moy[:,0], moy[:,1], 'b.')
    plt.plot(kp[:,0], kp[:,1], 'r.')
    
def train_cascade():

    start_time = time.time() # calculer le temps d'execution
    
    # charger images et points caracteristiques 
    image_set = np.load("mydataset/image_train_set.npy")
    landmarks_set = np.load("mydataset/landmarks_train_set.npy")

    n_samp, n_pts, dim = landmarks_set.shape
    landmarks_set = landmarks_set.reshape((n_samp, n_pts*dim)) # redimensionner

    model_moyen = np.load(path_train+"train_pts_moyens.npy") # charger modele moyen
    model_moyen = np.repeat(model_moyen[np.newaxis, :, :], image_set.shape[0], axis=0) # repeter le moyen

    landmarks_set_aug = landmarks_set
    
    # angmenter le dataset
    for i in range(1,11):
        perturb = np.load(path_train+str(i)+"_perturbation.npy")
        perturb = np.repeat(perturb[np.newaxis, :, :], image_set.shape[0], axis=0)
        model_moyen = np.concatenate((model_moyen, perturb),axis=0)
        landmarks_set_aug = np.concatenate((landmarks_set_aug, landmarks_set),axis=0)

    print(landmarks_set_aug.shape, model_moyen.shape, image_set.shape)
    print("-------> dataset augmented preprared!")

    n_iter = 5
    myCascade = Cascade()
    myCascade.fit(image_set, model_moyen, landmarks_set_aug, n_iter=n_iter)
    # myCascade.fit2(image_set, model_moyen, landmarks_set_aug, n_iter=n_iter)
    myCascade.save_model() # sauvegarder R et A

    print("time taken:", (time.time() - start_time)/60) # temps d'execution

def test_cascade():
    #charger images
    image_set = np.load("mydataset/image_test_set.npy")
    landmarks_set = np.load("mydataset/landmarks_test_set.npy")
    n_samp = image_set.shape[0]
    
    # charger l'initialisation de points
    pt_moyen = np.load(path_train+"train_pts_moyens.npy")
    n_pts, dim = pt_moyen.shape
    pts_moyen = np.repeat(pt_moyen[np.newaxis, :, :], image_set.shape[0], axis=0)
    Qs_star = landmarks_set - pts_moyen

    # charger R et A
    R = np.load("mydataset/R.npy")
    A = np.load("mydataset/A.npy")
    sift = cv2.xfeatures2d.SIFT_create()
    n_samp, n_pts, dim = pts_moyen.shape

    Qs_test = 0 # deplacement, initialise a 0
    for i in range(len(R)):
        
        R1 = R[i]
        A1 = A[i]
        
        X0_test = descriptor(pts_moyen, image_set, sift) # descripteur, entree du regresseur
        X0_test1 = X0_test.dot(A1.T) # la reduction par PCA
        
        Y1 = np.concatenate((np.ones((X0_test1.shape[0],1)), X0_test1), axis=1) # calcul de Y
        Qs_test1 = Y1.dot(R1) # la sortie du regresseur

        keypoints = pts_moyen.reshape((n_samp, n_pts*dim))
        keypoints = keypoints + Qs_test1 # nouvel emplacements des points
        Qs_test += Qs_test1
        
        keypoints = keypoints.reshape((pts_moyen.shape))
        drawKeypoints(image_set[1], pts_moyen[1], keypoints[1]) # visualiser
        pts_moyen = keypoints.reshape((n_samp, n_pts, dim)) # update moyen

        print("========> Iteration ", str(i+1))
    
        Qs_star = Qs_star.reshape(Qs_test.shape)
        err = mean_squared_error(Qs_test, Qs_star)
        print('Mean Squared Error :', err)

# train_cascade()
test_cascade()

