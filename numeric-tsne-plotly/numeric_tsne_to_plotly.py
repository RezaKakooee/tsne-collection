# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:45:28 2020

@author: reza
"""
#%%
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from settings import Params
from loader import load_data
from plotlyplots import plotting
from calculate_similarity import get_similarity

#%%
if __name__ == '__main__':
    # directory and file names
    params = Params()
    dataset_dir = params.dataset_dir
    log_dir = params.log_dir
    tsne_n_components = params.tsne_n_components
    tsne_perplexity = params.tsne_perplexity
    
    # load the whole image set
    num_samples, num_features, features, labels = load_data()

    # similarities
    similarity_mat = get_similarity(features, features)
    
    
    # tSNE 
    tsne = TSNE(n_components=tsne_n_components, perplexity=tsne_perplexity, random_state=0) #, metric='precomputed')
    tsne_val = tsne.fit_transform(features) #tsne.fit_transform(1.0001-similarity_mat)
    
    ### MPL
    plotting(tsne_val, labels)
