#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches


path = "300w/"
path_train_image = "300w/300w_train_images.txt"
path_train_lm = "300w/300w_train_landmarks.txt"
path_test_image = "300w/helen_testset.txt"
path_test_lm = "300w/helen_testset_landmarks.txt"



def load_images(path, datalist):
    """ load image files
            path: database path
            datalist: the list containing all the images path
        return:
            dataset: dataset of images
    """
    dataset = []
    for line in datalist:
        im = Image.open(path+line)
        dataset.append(im)
    return dataset

def load_landmarks(path, datalist):
    """ load face landmarks files
            path: database path
        return:
            datalist: the list containing all the landmarks files path
    """
    dataset = []
    for line in datalist:
        lm = np.loadtxt(path+line)
        lm = np.array(lm)
        dataset.append(lm)
    return np.array(dataset)


def load_train():
    """ save train dataset files paths in a list
        return:
            train_ims: train images file paths list
            train_lms: train landmarks file paths list
    """
    train_ims = []
    train_lms = []
    fo = open(path_train_image, "r")
    for line in fo.readlines(): 
        line = line.strip()
        train_ims.append(line)

    fo = open(path_train_lm, "r")
    for line in fo.readlines(): 
        line = line.strip()
        train_lms.append(line)
    
    
    return train_ims, train_lms

def load_test():
    """ save test dataset files paths in a list
        return:
            test_ims: test images file paths list
            test_lms: test landmarks file paths list
    """
    test_ims = []
    test_lms = []
    fo = open(path_test_image, "r")
    for line in fo.readlines(): 
        line = line.strip()
        test_ims.append(line)


    fo = open(path_test_lm, "r")
    for line in fo.readlines(): 
        line = line.strip()
        test_lms.append(line)  
    
    return test_ims, test_lms



def prepare_dataset(save_path, data, lms):
    """ generate dataset using the @data and @lms and save it in @save_path 
    """
    print("------------------> preparing my dateset...")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for n in range(len(data)):
        im = data[n]
        # fig1,ax1 = plt.subplots(1)
        # ax1.imshow(im)
               
        lm = lms[n]
        # ax1.plot(lm[:,0],lm[:,1],'r.')
        length = np.max(lm[:,0]) - np.min(lm[:,0])
        width = np.max(lm[:,1]) - np.min(lm[:,1])
        
        length = 1.3 * length
        width = 1.3 * width
        
        # calculer la boite englobante elargie 30%
        left, top = np.min(lm[:,0])- 0.15*length, np.min(lm[:,1])-0.15*width
        right, bottom = np.min(lm[:,0]) + 0.85*length, np.min(lm[:,1]) + 0.85*width
        
        # visualisation de rectangle
        # rect = patches.Rectangle((np.min(lm[:,0])- 0.15*length, np.min(lm[:,1])-0.15*width), length, width, linewidth=2,edgecolor='r',fill=None, alpha=1)
        # ax1.add_patch(rect)
        # plt.show()
        
        # cropper et re dimensionner les images
        im1 = im.crop((left, top, right, bottom)) 
        w,h = im1.size
        # fig2, ax2 = plt.subplots()
        im1 = im1.resize((128,128), Image.ANTIALIAS)
        # plt.imshow(im1)
        
        im1.save(save_path+str(n)+".jpg")
        
        # calculer les points caracteristiques correspondants aux nouvelles images
        lm[:,0] = lm[:,0] - left
        lm[:,1] = lm[:,1] - top
        
        factor = [128/w, 128/h]
        lm[:,0] = lm[:,0] * factor[0]
        lm[:,1] = lm[:,1] * factor[1]
        
        np.save(save_path+str(n)+".npy", lm)
        # ax2.plot(lm[:,0],lm[:,1],'r.')
    print("------------------> my dateset prepared successfully!")
 
def calculate_moyen(path, lms):
    """ calculate mean landmarks and save it in @path
        return:
            plm: mean landmarks
    """
    n_lms = len(lms)
    n_pts = len(lms[0])
    pts = np.zeros([n_lms, n_pts, 2])
    for n in range(len(lms)):
        pt = np.load(path+str(n)+".npy")
        pts[n,:,:] = pt
        
    plm = np.mean(pts, axis=0)
    print("------------------> landmarks mean calculated successfully!")
    return plm

def perturbations(path, pts_moyens):
    """ calculate 10 mean landmarks perturbed and save it in @path
    """
    translation = np.round((np.random.rand(2,10)-0.5)*20) # random translation +-20pixels
    echelle = (np.random.rand(2,10)-0.5) * 0.2 # random factor +- 0.2
    for i in range(10):
        lm = np.copy(pts_moyens)
        lm[:,0] = lm[:,0] - translation[0,i]
        lm[:,1] = lm[:,1] - translation[1,i]
        
        lm[:,0] = lm[:,0] * (echelle[0,i] + 1)
        lm[:,1] = lm[:,1] * (echelle[1,i] + 1)
        np.save(path+str(i+1)+"_perturbation.npy", lm)



def visualise_mydataset(path_train, path_test):
    pt_moyen_train = np.load(path_train+"train_pts_moyens.npy")
    pt_moyen_test = np.load(path_test+"test_pts_moyens.npy")
    
    fig1,ax1 = plt.subplots()
    im1 = Image.open(path_test+str(5)+".jpg")
    ax1.imshow(im1)
    ax1.plot(pt_moyen_test[:,0], pt_moyen_test[:,1], 'b.')
    
    fig2,ax2 = plt.subplots()
    im2 = Image.open(path_train+str(5)+".jpg")
    pt_moyen = np.load(path_train+str(5)+".npy")
    ax2.imshow(im2)
    ax2.plot(pt_moyen[:,0], pt_moyen[:,1], 'b.')
    
    im = Image.open(path_train+str(5)+".jpg")
    for i in range(1,11):
        fig,ax = plt.subplots()
        ax.imshow(im)
        pt_moyen_p = np.load(path_train+str(i)+"_perturbation.npy")
        ax.plot(pt_moyen_p[:,0], pt_moyen_p[:,1], 'r.')





train_im, train_lm = load_train() 
test_im, test_lm = load_test()
# print(len(train_im), len(test_im))

# images et points caracteristiques de la base originale d'apprentissage et de test
data_train_ims = load_images(path, train_im)
data_test_ims = load_images(path, test_im)
data_train_lms = load_landmarks(path, train_lm)
data_test_lms = load_landmarks(path, test_lm)
print(data_train_ims.shape, data_train_lms.shape)
print("------------------> date loaded successfully!")

path_train = "mydataset/train/"
path_test = "mydataset/test/"

# generer nouvelle base d'apprentissage
prepare_dataset(path_train, data_train_ims, data_train_lms)
pt_moyen_train = calculate_moyen(path_train, data_train_lms)
np.save(path_train+"train_pts_moyens.npy", pt_moyen_train)

# generer perturbations
pts_moyen = np.load(path_train+"train_pts_moyens.npy")
perturbations(path_train, pts_moyen)

# generer nouvelle base de test
prepare_dataset(path_test, data_test_ims, data_test_lms)
pt_moyen_test = calculate_moyen(path_test, data_test_lms)
np.save(path_test+"test_pts_moyens.npy", pt_moyen_test)

# decommenter pour visualiser my dataset
# visualise_mydataset(path_train, path_test)
