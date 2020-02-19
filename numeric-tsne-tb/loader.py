# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:53:07 2020

@author: rkako
"""
import pickle
import numpy as np
from settings import Params

params = Params()
data_path = params.data_path
    
def load_data():
    with open(data_path, "rb") as vectors_file:
        pca_vectors = pickle.load(vectors_file)
    
    features = pca_vectors[:1000]
    
    num_samples, num_features = np.shape(features)
    
    labels = np.random.randint(0, params.num_clusters, num_samples)
        
    return num_samples, num_features, features, labels

