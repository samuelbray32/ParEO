#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 08:55:12 2021

@author: sam
"""

import time
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.framework import ops
import pickle
import keras
from keras import layers
from keras import backend as K

#def M_project(ind):
#    #returns a measurement layer that is the projection of the index(s)
#    M = keras.layers.Lambda(lambda x: tf.expand_dims(x[:,ind],-1))
#    return M

def M_project(ind):
    #returns a measurement layer that is the projection of the index(s)
    M = keras.layers.Lambda(lambda x: tf.expand_dims(tf.gather(K.transpose(x),ind),-1))
    return M

def M_project_exponential(ind):
    #returns a measurement layer that is the projection of the index(s)
    M = keras.layers.Lambda(lambda x: tf.expand_dims(tf.gather(K.transpose(tf.exp(x)),ind),-1))
    return M

def M_project_shift(ind):
    #returns a measurement layer that is the projection of the index(s)
    M = keras.layers.Lambda(lambda x: tf.expand_dims(tf.gather(K.transpose(x+3),ind),-1))
    return M

def firingRate_project(ind):
    #rectifies state to firing rate and returns sum of connected neurons
    if type(ind) is int:
        M = keras.layers.Lambda(lambda x: tf.expand_dims(tf.gather(K.transpose(tf.nn.relu(x)),ind),-1))
    else:
        M = keras.layers.Lambda(lambda x: tf.reduce_sum(K.transpose(tf.gather(K.transpose(tf.nn.relu(x)),ind)),axis=-1,keep_dims=True))
    return M

class M_neuron(keras.layers.Layer):
    #Layer that rectifies inuts to firing rates and linear transforms to 
    #functional output (generalization of firingRate_project)
    def __init__(self,n, n_out=1, W_out=None,trainable=True,initializer=tf.initializers.glorot_uniform(),
                 **kwargs):
        super(M_neuron, self).__init__(**kwargs)
        if W_out is None:
            self.W_out = self.add_weight(name='W_out', 
                                    shape=(n,n_out), 
                                    initializer=initializer,
                                    trainable=trainable,)
        else:
            self.W_out = self.add_weight(W_out,trainable=trainable,name='W_out')
        self.n = n
        self.n_out = n_out
    
    def build(self, input_shape,):
        super(M_neuron, self).build(input_shape)
        
    def call(self,X):
        r = tf.nn.relu(X)
        print(self.W_out.shape,r.shape)
        Y = K.dot(r,self.W_out)
        print(Y.shape)
        return Y
    