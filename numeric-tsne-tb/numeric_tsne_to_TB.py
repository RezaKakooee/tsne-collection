#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from settings import Params
from loader import load_data
from embeder import numeric_embeding_creator
from calculate_similarity import get_similarity

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.contrib.tensorboard.plugins import projector


#%%
def get_data():
    return load_data()

def load_model():
    return InceptionV3(include_top=False, pooling='avg')

# get features
def get_deep_features(model, images):
    return model.predict(images)

def embeding_creator(numeric_data):
    numeric_embeding_creator(numeric_data)

def save_metadata(metadata_path, labels):
    with open(metadata_path,'w') as f: # "metadata.tsv"
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index,label))
                    
#%% Run
params = Params()
if __name__ == '__main__':
    ### directory and file names
    dataset_dir = params.dataset_dir
    log_dir = params.log_dir 
    metadata_path = params.metadata_path
    
    ### load the whole image set
    num_samples, num_features, features, labels = get_data()
    
    ### save metadata
    df_labels = pd.DataFrame(data={'Label':labels}, index=np.arange(num_samples))
    save_metadata(metadata_path, labels)
    
    ### model
#    model = load_model()
    
    ### get deep feature
#    deep_features = get_deep_features(model, features)
    
    ### similarities
    similarity_mat = get_similarity(features, features)
        
    ### tSNE 
    tsne = TSNE(n_components=3, perplexity=9.0, random_state=0) #, metric='precomputed')
    tsne_val = tsne.fit_transform(features)
    
    
    ### embeding
    embeding_images_with = params.embeding_images_with
    if  embeding_images_with == 'feat':
        embd_mat = features
    elif embeding_images_with == 'deepfeat':
        embd_mat = deep_features
    elif embeding_images_with == 'sim':
        embd_mat = similarity_mat 
    elif embeding_images_with == 'tsne':
        embd_mat = tsne_val
        
    numeric_embeding_creator(embd_mat)
    
    #tensorboard --logdir=logs

