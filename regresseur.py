# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 23:04:04 2020

@author: HASEE
"""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import PCA
import cv2



class Cascade:
    
    def __init__(self):

        self.R = []
        self.A = []
        self.e = []
        self.sift = cv2.xfeatures2d.SIFT_create()
        
    
    def predict(self, Y, R):
        return Y.dot(R)
    
    
    def calculate_R(self, Y, Qs):
        R = (inv(Y.T.dot(Y))).dot(Y.T).dot(Qs)
        print("----> R calculated")
        return R
    
    
    def reduction_pca(self, X):
        pca = PCA(n_components=0.98)
        pca.fit(X)
        X = pca.transform(X)
        A = pca.components_
        print("----> pca calculated")
        return X,A
    
    
    def oneStepAlignment(self, img, pts_opt, pts_pre):
        
        X = descriptor(pts_pre, img, self.sift)
        print(X.shape)
        pts_pre = pts_pre.reshape((pts_pre.shape[0], pts_pre.shape[1]*pts_pre.shape[2]))
        X_c, A = self.reduction_pca(X)
        print("A shape:", A.shape)
        Qs_star = pts_opt - pts_pre
        Y = np.concatenate((np.ones((X_c.shape[0],1)), X_c), axis=1)
        
        R = self.calculate_R(Y, Qs_star)
        print("R shape:", R.shape)
        
        Qs0 = self.predict(Y, R)
        err = mean_squared_error(Qs0, Qs_star)
        print('Mean Squared Error :', err)
        
        pts_new = self.majPoints(pts_pre, Qs0)
        print("-----> one step!")
        
        return pts_new, R, A, err
        
        
    
    
    def majPoints(self, Sk, Qsk):
        return Sk + Qsk
    
    def fit(self, img_set, moy, landmarks_set, n_iter=5):

        for _ in range(n_iter):
            pts_new, r, a, err = self.oneStepAlignment(img_set, landmarks_set, moy)
            moy = pts_new.reshape(moy.shape)
            self.R.append(r)
            self.A.append(a)
            self.e.append(err)
            
    def fit2(self,  img_set, moy, landmarks_set, n_iter=5):
        
        for _ in range(n_iter):
            X0 = descriptor(moy, img_set)
            X0, a = self.reduction_pca(X0)
            moy = moy.reshape((moy.shape[0], 68*2))

            Qs_star = landmarks_set - moy
            clf = LinearRegression()
            clf.fit(X0, Qs_star)
            R0 = clf.coef_
            b0 = clf.intercept_
            r = np.concatenate((b0, R0))
            
            Y = np.concatenate((np.ones((X0.shape[0],1)), X0), axis=1)
            Qs0 = self.predict(Y, r)
            moy = self.majPoints(moy, Qs0)
            
            self.R.append(r)
            self.A.append(a)
        
 
    def save_model(self):
        np.save("mydataset/R3.npy", np.array(self.R))
        np.save("mydataset/A3.npy", np.array(self.A))
        print(np.array(self.R).shape)


def descriptor(pts, img, sift):
    
    descr = []
    # after an iterration, the points deplacement has same elements as images
    for i in range(pts.shape[0]):
        descripteur = None
        im = img[i%img.shape[0]]
        bottom = int(max(128, max(pts[i,:,1])+20)) - 128
        right = int(max(128, max(pts[i,:,0])+20)) - 128

        for n in range(pts.shape[1]):
            keypoints = [cv2.KeyPoint(x=pts[i,n,0],y=pts[i,n,1], _size=20)]
            img1 = cv2.copyMakeBorder(im, 0, bottom, 0, right,  cv2.BORDER_REPLICATE,value=0)
            kp, des = sift.compute(img1, keypoints)
            if not isinstance(descripteur, np.ndarray):
                descripteur = des
            else:
                descripteur = np.concatenate((descripteur, des), axis=1)
                # print(descripteur.shape)
            
        descr.append(descripteur[0])

    print("----> descr calculated")
    return np.array(descr)
