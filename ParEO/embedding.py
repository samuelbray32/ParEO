#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 09:03:56 2021

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

def inferenceNetwork(context_dim=None, parameter_dim=None, 
                     merged_layers=[4,], parameter_layers=[4,],
                     latent=2, activation='relu',regular=None,
                     dropout=0,anchor_ind=-1,
                     square_output=False, conditioning=True):
    #inputs
    Xin = keras.layers.Input(shape=(context_dim,))
    Pin = keras.layers.Input(shape=(parameter_dim,))
    #conditioning network
    if conditioning:
        P = keras.layers.Dense(parameter_layers[0], activation = 'relu', kernel_regularizer=regular)(Pin)
        for l in parameter_layers[:-1]:
            P = keras.layers.Dense(l, activation = 'relu', kernel_regularizer=regular)(P)
        P = keras.layers.Dense(parameter_layers[-1], activation = 'linear')(P)
        #concatenate
        X = keras.layers.Concatenate()([Xin,P])
    else:
        X = Xin
    #do nonlinear functioning
    for l in merged_layers:
        X = keras.layers.Dense(l,activation=activation,kernel_regularizer=regular)(X)
        if dropout:
            X = keras.layers.Dropout(dropout)(X)
    #embed in latent dimension (same as your mechanistic model dimension)
    X = keras.layers.Dense(latent,activation='linear',)(X)
    #square so everythings positive
    if square_output:
        X = keras.layers.Lambda(lambda x: K.square(x))(X)
    model = keras.models.Model(inputs=[Xin,Pin],outputs=X)
    return model


def inferenceNetwork_RNN(context_dim=None, parameter_dim=None, 
                     postRNN_layers=[], parameter_layers=[4,],
                     latent=2, activation='relu',regular=None,
                     dropout=0,anchor_ind=-1,
                     square_output=False, stimulus=True, conditioning=True):
    #inputs
    Xin = keras.layers.Input(shape=(context_dim,))
    #split Xin into X and U, reshape
    if stimulus:
        U = keras.layers.Lambda(lambda x: x[:,:context_dim//2])(Xin)
        X = keras.layers.Lambda(lambda x: x[:,context_dim//2:])(Xin)
        X = keras.layers.Reshape((context_dim//2,1))(X)
        U = keras.layers.Reshape((context_dim//2,1))(U)
        X = keras.layers.Concatenate(axis=-1)([X,U])
    else:
        X = keras.layers.Reshape((context_dim,1))(Xin)
    print('X', X.shape)
    if conditioning:
        Pin = keras.layers.Input(shape=(parameter_dim,))
        #conditioning network
        P = keras.layers.Dense(parameter_layers[0], activation = 'relu', kernel_regularizer=regular)(Pin)
        for l in parameter_layers[1:-1]:
            P = keras.layers.Dense(l, activation = 'relu', kernel_regularizer=regular)(P)
        P = keras.layers.Dense(parameter_layers[-1], activation = 'linear')(P)
        #tile parameters
        if stimulus:
            samples = context_dim//2
        else:
            samples = context_dim
        P_feed = keras.layers.Lambda(lambda x: K.tile(x, samples))(P)
        P_feed = keras.layers.Reshape((samples,-1))(P_feed)
        #concatenate conditioning parameter onto each timestep
        X = keras.layers.Concatenate()([X,P_feed])
        print('X+P',X.shape)
    #apply RNN
    Q = keras.layers.GRU(units=latent)(X)
    if len(postRNN_layers)>0:
        for l in postRNN_layers:
            Q = keras.layers.Dense(l, activation='relu')(Q)
    Q = keras.layers.Dense(latent, activation='linear')(Q)
    #square so everythings positive
    if square_output:
        Q = keras.layers.Lambda(lambda x: K.square(x))(Q)
    #define model
    if conditioning:
        model = keras.models.Model(inputs=[Xin,Pin],outputs=Q)
    else:
        model = keras.models.Model(inputs=Xin,outputs=Q)
    return model