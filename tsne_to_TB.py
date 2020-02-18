#%%
import os
import numpy as np
import pandas as pd

from settings import Params
from make sprite import images_to_sprite

import matplotlib.pyplot as plt#, mpld3
from mpl_toolkits.mplot3d import Axes3D
from Annotate_Images_to3D import ImageAnnotations3D

from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.contrib.tensorboard.plugins import projector


#%% load model
def load_model():
    model = InceptionV3(include_top=False, pooling='avg')
    return model

# get features
def get_features(model, images):
    features = model.predict(images)
    return features

def embeding_creator(images, LOG_DIR, path_for_sprites, path_for_metadata):
    embedding_var = tf.Variable(images, name="image_embedding")
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = path_for_metadata
    embedding.sprite.image_path = path_for_sprites 
    embedding.sprite.single_image_dim.extend([IMG_W,IMG_W])
    projector.visualize_embeddings(summary_writer, config)

def save_model(LOG_DIR):
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)

##%%
def matplotting(tsne_val, imgs, flag=True, title='Recommender'):
    if flag==True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)
        xs = np.array(tsne_val[:,0]).reshape((-1,1))
        ys = np.array(tsne_val[:,1]).reshape((-1,1))
        zs = np.array(tsne_val[:,2]).reshape((-1,1))
        ax.scatter(xs, ys, zs, marker=".", s=0.1)
        ax2 = fig.add_subplot(111,frame_on=False) 
        ax2.axis("off")
        ax2.axis([0,1,0,1])
        ia = ImageAnnotations3D(np.c_[xs,ys,zs], imgs, ax, ax2)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
#        ax.set_title(title)
        plt.show()
        
#    return fig, ax
        
#%% Run
if __name__ == '__main__':
    # directory and file names
    dataset_name = 'logos'
    images_directory = "C:/Users/reza/gdrive-redu/hslu/HSLU-TSNE-Oct2019-Others/all-deal-images/"
    similarity_table_name='similarity_table.csv'
    similarity_directory = "C:/Users/Reza/gdrive-redu/hslu/HSLU-TSNE-Oct2019/"
    similarity_table_path = similarity_directory + similarity_table_name
    
    resim_flag = 'False' # True or False or resim
    
    LOG_DIR = 'minimalsample'
    path_for_sprites = "sprit.png"
    path_for_metadata = "metadata.tsv"
    
    # load the whole image set
    n_imgs, images_pathes, images_index, images_vec, images_mat, images_arr, labels, similarity_mat = load_images_and_similarity(images_directory, similarity_table_path)
    IMG_W = images_mat[0].shape[0]
       
    imgs_list = list(images_arr)
    imgs_arr = images_arr
    imgs_mat = images_mat

    # model
    model = load_model()
    
    # get feature
    features = get_features(model, imgs_arr)
    
    # similatiris
    similarity_mat_cos = cosine_similarity(features, features)
    
    distance_mat_euc = euclidean_distances(features, features)
    similarity_mat_euc = 1- normalize(distance_mat_euc)
    
    LOG_DIR_ = LOG_DIR
    for s in ['rec', 'cos', 'euc']:
        LOG_DIR = LOG_DIR_ + "\\" + s + "\\"
        ### save sprit images
        plt.imsave(LOG_DIR + "\\" + path_for_sprites, sprite_image, cmap='gray')
        
        if s == 'rec':
            similarity_mat_rec = similarity_mat
        elif s == 'cos':
            similarity_mat = similarity_mat_cos
        elif s == 'euc':
            similarity_mat = similarity_mat_euc
                
        # embeding
        embeding_with = 'sim'# 'sim'# 'feat'# 'img'# 
        if  embeding_with == 'img':
            embd_mat = imgs_arr
        elif embeding_with == 'feat':
            embd_mat = features
        elif embeding_with == 'sim':
            embd_mat = similarity_mat 
        embeding_creator(embd_mat, LOG_DIR, path_for_sprites, path_for_metadata)
        
        # save model
        with open(LOG_DIR + "\\" + path_for_metadata,'w') as f: # "metadata.tsv"
            f.write("Index\tLabel\n")
            for index, label in enumerate(labels):
                f.write("%d\t%d\n" % (index,label))
        save_model(LOG_DIR)
    
    # tSNE 
    distance_matrix_rec = 1.000001-similarity_mat_rec # rec = recommander
    distance_matrix_cos = 1.000001-similarity_mat_cos
    distance_matrix_euc = distance_mat_euc
   
    tsne = TSNE(n_components=3, perplexity=8.0, random_state=0, metric='precomputed')
    tsne_val_rec = tsne.fit_transform(distance_matrix_rec)
    tsne_val_cos = tsne.fit_transform(distance_matrix_cos)
    tsne_val_euc = tsne.fit_transform(distance_matrix_euc)
    
    tsne_val = [tsne_val_rec, tsne_val_cos, tsne_val_euc]
    titles = ['Recommender', 'Cosine', 'Euclidean']
    
    ### MPL
    for each_tsne_val, title in zip(tsne_val, titles):
        matplotting(each_tsne_val, imgs_list, flag=True, title=title)

    #tensorboard --logdir=rec
    #tensorboard --logdir=cos
    #tensorboard --logdir=euc

