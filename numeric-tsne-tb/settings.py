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
        self.dataset_dir = 'C:/Users/rkako/gdrive-edu/hslu/HSLU-Secude/large-data'
        self.data_file_name = 'pca_output_vectors.pickle'
        self.data_path = os.path.join(self.dataset_dir, self.data_file_name)
        
        self.log_dir_name = 'logs'
        self.log_dir = os.path.join(self.current_dir, self.log_dir_name)
        self.sprit_image_name = "sprit.png"  
        self.sprit_image_path =  os.path.join(self.log_dir, self.sprit_image_name)
        self.metadata_name = "metadata.tsv"
        self.metadata_path =  os.path.join(self.log_dir, self.metadata_name)

        # Images
        self.similarity_metric = 'cosine'
        
        # clucters
        self.num_classes = 10
        
        # embeder
        self.embeding_images_with = 'tsne'# 'sim'# 'feat'# 'img'# 
                
#params = Params()