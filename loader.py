# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:53:07 2020

@author: rkako
"""
import os
import numpy as np
import pandas as pd
from settings import Params
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

params = Params()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def load_images_and_similarity():
    # get image names
    images_names = os.listdir(params.imageset_dir)
    images_indecis = [img_name.replace('logo_', '') for img_name in images_names]
    images_indecis_str = [img_ind.replace('.png', '') for img_ind in images_indecis]
    images_indecis_int = [int(img_ind_str) for img_ind_str in images_indecis_str]
                          
    # get similar image names
    similarity_table = pd.read_csv(params.similarity_table_path, delimiter=';')
    col1 = similarity_table[params.column1_name]
    col2 = similarity_table[params.column2_name]
    cols = np.array([col1.values, col2.values]).flatten()    
    uniq_images_name = np.unique(cols, return_index=False)#[:20]
    
    # finding accepted images
    accepted_images = set(images_indecis_int).intersection(set(uniq_images_name))
    accepted_images_names = list(accepted_images)
    n_accepted_images_names = len(accepted_images_names)
    # make similarity matrix
    similarity_matrix = np.identity(n_accepted_images_names)
    for a, b, s in similarity_table.values:
        i, j = np.where(accepted_images_names==a), np.where(accepted_images_names==b)
        similarity_matrix[i, j] = s
        similarity_matrix[j, i] = s
    
    # make image pathes
    image_name_format = "logo_{}.png"
    images_pathes = [params.imageset_dir + image_name_format.format(acc_img_name) for acc_img_name in accepted_images_names]
    
    # convert images to other formats
    images_arr = []
    img_shapes = []
    for img_path in images_pathes:
        img = load_img(img_path, target_size=(params.img_targ_H, params.img_targ_W))
        img = img_to_array(img)
        img /= 255.0
        images_arr.append(img)
        img_shapes.append(img.shape)
        
    images_arr = np.array(images_arr)
    images_mat = np.array([rgb2gray(img) for img in images_arr])
    images_vec = images_mat.reshape(n_accepted_images_names, params.img_targ_H*params.img_targ_W)
    labels = np.arange(0, n_accepted_images_names)
    
    return n_accepted_images_names, images_pathes, accepted_images_names, images_vec, images_mat, images_arr, labels, similarity_matrix

load_images_and_similarity()