# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:38:13 2020

@author: reza
"""
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from settings import Params

params = Params()
log_dir = params.log_dir
metadata_path = params.metadata_path

def numeric_embeding_creator(numeric_data):
    tf_data = tf.Variable(numeric_data, name="numeric_embedding")
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver([tf_data])
    saver.save(sess, os.path.join(log_dir, "model.ckpt"), 1)
    
    summary_writer = tf.summary.FileWriter(log_dir)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = tf_data.name
    embedding.metadata_path = metadata_path
    projector.visualize_embeddings(summary_writer, config)