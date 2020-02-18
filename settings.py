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
        self.imageset_folder_name = 'imageset'
        self.imageset_dir = os.path.join(self.current_dir, self.imageset_folder_name)
        self.similarity_folder_name = 'precomputed-similarity'
        self.similarity_directory = os.path.join(self.current_dir, self.similarity_folder_name)
        self.similarity_table_name = 'similarity_table.csv'        
        self.similarity_table_path = os.path.join(self.similarity_directory, self.similarity_table_name)
        
        self.resim_flag = 'False' # True or False or resim
        
        self.log_dir = 'logs'
        self.sprit_image_name = "sprit.png"  
        self.sprit_image_path =  os.path.join(self.current_dir, self.sprit_image_name)
        self.path_for_metadata = "metadata.tsv"
        self.smetadata_path =  os.path.join(self.current_dir, self.sprit_image_name)

        # Images
        self.img_targ_H = 299
        self.img_targ_W = 299
        
        # Similarity Matrix 
        self.column1_name = 'item1' 
        self.column2_name = 'item2'
        
params = Params()