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
                
        self.log_dir_name = 'logs'
        self.log_dir = os.path.join(self.current_dir, self.log_dir_name)
        self.sprit_image_name = "sprit.png"  
        self.sprit_image_path =  os.path.join(self.log_dir, self.sprit_image_name)
        self.metadata_name = "metadata.tsv"
        self.metadata_path =  os.path.join(self.log_dir, self.metadata_name)

        # Images
        self.img_targ_H = 299
        self.img_targ_W = 299
        self.embeding_images_with = 'sim'# 'sim'# 'feat'# 'img'# 
        self.similarity_metric = 'cosine'
                
#params = Params()