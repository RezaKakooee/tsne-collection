# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:07:26 2020

@author: reza
"""
from settings import Params
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

params = Params()
def get_similarity(features1, features2):
    if params.similarity_metric == 'cosine':
         similarity_mat = cosine_similarity(features1, features2)  
    elif params.similarity_metric == 'euclidian':
         distance_mat_euc = euclidean_distances(features1, features2)
         similarity_mat = 1- normalize(distance_mat_euc)
    
    return similarity_mat