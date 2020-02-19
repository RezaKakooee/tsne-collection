# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:13 2020

@author: reza
"""
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from settings import Params

params = Params()
log_dir = params.log_dir
sprite_image_path = params.sprite_image_path
metadata_path = params.metadata_path
image_height = params.image_height
image_width = params.image_width

def image_embeding_creator(images):
    embedding_var = tf.Variable(images, name="image_embedding")
    summary_writer = tf.summary.FileWriter(log_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = metadata_path
    embedding.sprite.image_path = sprite_image_path
    embedding.sprite.single_image_dim.extend([image_height, image_width])
    projector.visualize_embeddings(summary_writer, config)
