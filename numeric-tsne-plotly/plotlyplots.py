# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:59:13 2020

@author: reza
"""
import numpy as np
import pandas as pd
import plotly.express as px

def plotting(tsne_val, labels):
    xs = list(tsne_val[:,0]) #np.array(tsne_val[:,0]).reshape((-1,1))
    ys = list(tsne_val[:,1]) #np.array(tsne_val[:,1]).reshape((-1,1))
    zs = list(tsne_val[:,2]) #np.array(tsne_val[:,2]).reshape((-1,1))
    ls = list(labels)
    df = pd.DataFrame(data={'X_values': xs, 'Y_values': ys, 'Z_values': zs, 'c_values': ls})
    
    fig = px.scatter_3d(df, x='X_values', y='Y_values', z='Z_values', color='c_values')
    fig.show()
    
        