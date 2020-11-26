# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:22:32 2020

@author: HASEE
"""
import os
import numpy as np
import cv2
from regresseur import descriptor
import matplotlib.pyplot as plt

path_img = "mydataset/mydataset/" # path de mes images
path_train = "mydataset/train/" # path de train pour avoir le modele moyen
pt_moyen = np.load(path_train+"train_pts_moyens.npy") # cahrger le modele poyen

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
landmarks_set = []
image_set = []

for root, dirs, files in os.walk(path_img, topdown=False):
    x0 = []
    i = 0
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(path_img + file) # lire une image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convertir en gris
            faces = face_cascade.detectMultiScale(gray, 1.3, 1) # detecter si il a un visage
            if faces != (): # si il y en a
                for [x,y,w,h] in faces:
                    if w < gray.shape[0]/5: # seuiller pour abandonner les faux visages
                        continue
                    
                    # agrandir la boite englobante de 30%
                    x, y = int(x - 0.15*w), int(y - 0.15*h)
                    w = int(1.3 * w)
                    h = int(1.3 * h)
                    gray = gray[y:y+h, x:x+w]
                    if gray.shape > (10,10): # si l'image est bien cropee
                        gray = cv2.resize(gray, (128, 128))
                        image_set.append(gray)
                
            if i == 20: # si beaucoup d'images, on teste que sur 20 premieres 
                break
            i += 1

                
image_set = np.array(image_set)
print("image set est de dimension ", image_set.shape)

def drawKeypoints(img, moy, kp):
    plt.figure()
    plt.imshow(img)
    plt.plot(moy[:,0], moy[:,1], 'b.')
    plt.plot(kp[:,0], kp[:,1], 'r.')
    
    
def test_cascade(image_set, pt_moyen):

    n_samp = image_set.shape[0]
    n_pts, dim = pt_moyen.shape
    pts_moyen = np.repeat(pt_moyen[np.newaxis, :, :], image_set.shape[0], axis=0)

    # charger R et A
    R = np.load("mydataset/R3.npy")
    A = np.load("mydataset/A3.npy")
    
    # descripteur
    sift = cv2.xfeatures2d.SIFT_create()
    n_samp, n_pts, dim = pts_moyen.shape

    for i in range(len(R)):
        
        R1 = R[i]
        A1 = A[i]
        
        X0_test = descriptor(pts_moyen, image_set, sift)# descripteur, entree du regresseur
        X0_test1 = X0_test.dot(A1.T)# la reduction par PCA
        Y1 = np.concatenate((np.ones((X0_test1.shape[0],1)), X0_test1), axis=1)# calcul de Y
        Qs_test1 = Y1.dot(R1)# la sortie du regresseur
        
        keypoints = pts_moyen.reshape((n_samp, n_pts*dim))
        keypoints = keypoints + Qs_test1# nouvel emplacements des points
        
        keypoints = keypoints.reshape((pts_moyen.shape))
        i1, i2, i3 = 6, 10, 7
        drawKeypoints(image_set[i1], pts_moyen[i1], keypoints[i1])
        drawKeypoints(image_set[i2], pts_moyen[i2], keypoints[i2])
        drawKeypoints(image_set[i3], pts_moyen[i3], keypoints[i3])
        pts_moyen = keypoints.reshape((n_samp, n_pts, dim))# update moyen

        print("========> Iteration ", str(i+1))


test_cascade(image_set, pt_moyen)
