# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:53:07 2020

@author: rkako
"""
import os
import numpy as np
from settings import Params
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

params = Params()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def get_labels(image_names, num_images):
    # Labels for coil-5
    uniqe_labels_str = np.unique([image_name[3:5] for image_name in image_names])
    len_uniqe_labels_str = len(uniqe_labels_str)
    unique_labels = np.arange(len_uniqe_labels_str)
    labels = np.zeros(num_images)
    j = 0
    for i in range(0,num_images,5):
        labels[i:i+5] = unique_labels[j]
        j += 1
    return labels
    
def load_images():
    # get image names
    image_names = os.listdir(params.imageset_dir)
    num_images = len(image_names)
    # make image pathes
    image_pathes = [os.path.join(params.imageset_dir, img_name) for img_name in image_names]
    
    # convert images to other formats
    images_arr = []
    img_shapes = []
    for img_path in image_pathes:
        img = load_img(img_path, target_size=(params.img_targ_H, params.img_targ_W))
        img = img_to_array(img)
        img /= 255.0
        images_arr.append(img)
        img_shapes.append(img.shape)
        
    images_arr = np.array(images_arr)
    images_mat = np.array([rgb2gray(img) for img in images_arr])
    images_vec = images_mat.reshape(num_images, params.img_targ_H*params.img_targ_W)
    images_list = list(images_arr)
    
    labels = get_labels(image_names, num_images)
    
    return num_images, image_pathes, image_names, images_vec, images_list, images_mat, images_arr, labels

#num_images, image_pathes, image_names, images_vec, images_mat, images_arr, labels = load_images()