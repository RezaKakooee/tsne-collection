# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:53:36 2020

@author: rkako
"""

import os


class Params():
    def __init__(self):
        # Directories ans Pathes
        self.current_dir = os.getcwd()
#        self.dataset_folder_name = 'imageset'
        self.dataset_dir = 'C:/Users/reza/gdrive-redu/hslu/HSLU-Secude/large-data'
        self.data_file_name = 'pca_output_vectors.pickle'
        self.data_path = os.path.join(self.dataset_dir, self.data_file_name)
        
        self.log_dir_name = 'logs'
        self.log_dir = os.path.join(self.current_dir, self.log_dir_name)

        self.similarity_metric = 'cosine'
        
        # clucters
        self.num_clusters = 10
        
        # TSNE
        self.tsne_n_components = 3
        self.tsne_perplexity = 5.0
#params = Params()