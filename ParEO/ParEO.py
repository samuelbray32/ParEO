#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:54:29 2021

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
from .measurement import M_project   
    
class theta_layer(keras.layers.Layer):
    ## Layer that calls the THETA model
    def __init__(self,theta,initializer='ones',logspace=False,**kwargs):
        super(theta_layer, self).__init__(**kwargs)        
        #Set the dynamics update model
        self.theta = theta
        #Set the generic parameter variable
        self.P = self.add_weight(name='Parameters', 
                                    shape=(len(self.theta.param_names()),), 
                                    initializer=initializer,
                                    trainable=True)
        #whether to run training in log space to prevent negative flip and improve small value extimates
        self.logspace=logspace
        

    def build(self, input_shape,):
        super(theta_layer, self).build(input_shape)
        
    def call(self, X, **kwargs):
#        return self.THETA.dynamics(X, self.P, **kwargs) 
        if self.logspace:
            Qf = self.theta.dynamics(X, tf.exp(self.P), **kwargs)
        else:
            Qf = self.theta.dynamics(X, self.P, **kwargs) 
        return Qf
    
    def param_names(self):
        return self.theta.param_names()
        
#    def compute_output_shape(self, input_shape):
#        return self.P.shape
#    
#    def P(self,):
#        return self.get_weights()[0]


    
def ParEO_network(N, theta, batch_size, tau_pred, dt, input_shape,
                  M=M_project(0), initializer='ones',
                  lr=1e-3, beta1=1e0, beta2=1e1, beta3=3e1, input_d=1, method='taylor',
                  logspace=False, loss='mae', single_step=False, optimizer=None):
    
    if type(N) is str:
        N = keras.models.load_model(N,)# custom_objects={'arctan_relative_loss':arctan_relative_loss})
    Xin = keras.layers.Input(dtype='float',shape=input_shape)
    Yin = keras.layers.Input(dtype='float',shape=input_shape)
    if input_d==1:
        Uin = keras.layers.Input(dtype='float',shape=(tau_pred,))
    else:
        Uin = keras.layers.Input(dtype='float',shape=(tau_pred,input_d))
    THETA =  theta_layer(theta, initializer=initializer, logspace=logspace)
    Q = N(Xin)
    Y_t = M(Q)
    Q_fut_hat = THETA(Q,u=Uin,dt=dt)
    Y_fut_hat = M(Q_fut_hat)
    Q_fut = N(Yin)
    
    if loss=='mae':
        def loss_wrapper(Y, Q_fut_hat, Q_fut):
            def loss_f(y_true,y_pred):
                reconstruction = beta1*tf.reduce_mean(keras.losses.mean_absolute_error(Y,Xin[:,-1]))
                latent_pred = beta2*tf.reduce_mean(keras.losses.mean_absolute_error(Q_fut, Q_fut_hat))
                pred = beta3*tf.reduce_mean(keras.losses.mean_absolute_error(y_true,y_pred))
                return reconstruction + latent_pred + pred
            return loss_f
    elif loss=='mse':
        def loss_wrapper(Y, Q_fut_hat, Q_fut):
            def loss_f(y_true,y_pred):
                reconstruction = beta1*tf.reduce_mean(keras.losses.mean_squared_error(Y,Xin[:,-1]))
                latent_pred = beta2*tf.reduce_mean(keras.losses.mean_squared_error(Q_fut, Q_fut_hat))
                pred = beta3*tf.reduce_mean(keras.losses.mean_squared_error(y_true,y_pred))
                return reconstruction + latent_pred + pred
            return loss_f
    elif loss=='exponential':
        def loss_wrapper(Y, Q_fut_hat, Q_fut):
            def loss_f(y_true,y_pred):
                reconstruction = beta1*tf.reduce_mean(-tf.math.log(1/Y)+(1/Y)*Xin[:,-1])
                latent_pred = beta2*tf.reduce_mean(keras.losses.mean_squared_error(Q_fut, Q_fut_hat))
                pred = beta3*tf.reduce_mean(-tf.math.log(1/y_pred)+(1/y_pred)*y_true[:,-1])
                return reconstruction + latent_pred + pred
            return loss_f
        
    model = keras.models.Model([Xin,Yin,Uin], Y_fut_hat)
    if optimizer is None:
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,loss=loss_wrapper(Y_t,Q_fut_hat,Q_fut), ) 
    par_layers = {'Q': Q,
              'Y_t': Y_t,
              'Q_fut_hat': Q_fut_hat,
              'Y_fut_hat': Y_fut_hat,
              'Q_fut': Q_fut,
              'Xin': Xin,
              'Yin': Yin,
              'Uin': Uin,
              'THETA':  THETA}
    return model, par_layers

def EM_fit_pareo(model,input_data,output_data,
                 warmup=10,epochs_flip=1,EM_rounds=20,
                 batch_size=128):
    if warmup>0:
        print('WARMUP')
        model.fit(x=input_data,y=output_data,epochs=warmup,
                  batch_size=batch_size,verbose=1,shuffle=True,)
    E_history=[]
    M_history=[]
    #make model for E-step   
    Emodel = keras.models.Model(model.input,model.output)
    for l in Emodel.layers[1].layers:
        l.trainable=True
        #freeze THETA
        Emodel.layers[2].trainable = False
    Emodel.compile(optimizer=model.optimizer,loss=model.loss)
    #make model for M-step
    Mmodel = keras.models.Model(model.input,model.output)
    for l in Mmodel.layers[1].layers:
        l.trainable=True
        #freeze THETA
        Mmodel.layers[2].trainable = False
    Mmodel.compile(optimizer=model.optimizer,loss=model.loss)
    #run EM
    for i in range(EM_rounds):
        #E-step
        print(f'E-step {i}')
        Emodel.fit(x=input_data,y=output_data,epochs=epochs_flip,
                  batch_size=batch_size,verbose=1,shuffle=True,)
        E_history.extend(Emodel.history.history['loss'])
        #M-step
        print(f'M-step {i}')
        Mmodel.fit(x=input_data,y=output_data,epochs=epochs_flip,
                  batch_size=batch_size,verbose=1,shuffle=True,)
        M_history.extend(Mmodel.history.history['loss'])
    return E_history,M_history
    



#def ParEO_network_prior(N, theta, batch_size, tau_pred, dt, s,
#                        M=M_project(0), initializer='ones',
#                  lr=1e-3, beta1=1e0, beta2=1e1, beta3=3e1, input_d=1,
#                  logspace=False, loss='mae', single_step=False, method='taylor', optimizer=None,
#                  prior=None, beta_prior=1e0, prior_regularizer='L1', prior_logspace=False):
#    if type(N) is str:
#        N_i = keras.models.load_model(N, custom_objects={'arctan_relative_loss':arctan_relative_loss})
#    else:
#        N_i = N
#    Xin = keras.layers.Input(dtype='float',shape=(s,))
#    Yin = keras.layers.Input(dtype='float',shape=(s,))
#    if input_d==1:
#        Uin = keras.layers.Input(dtype='float',shape=(tau_pred,))
#    else:
#        Uin = keras.layers.Input(dtype='float',shape=(tau_pred,input_d))
#    print (Xin.shape, Yin.shape, Uin.shape)
#    pareo =  ParEO(N_i, theta, batch_size, initializer=initializer, logspace=logspace)
#    Q = pareo(Xin, dynamics=False, project=False, tau=tau_pred, dt=dt)
#    Y_t = M(Q)#pareo(Xin, dynamics=False, project=True, tau=tau_pred, dt=dt)
#    Q_fut_hat = pareo(Xin, dynamics=True, project=False, u=Uin, tau=tau_pred, dt=dt, single_step=single_step, method=method)
#    Y_fut_hat = M(Q_fut_hat)#pareo(Xin, dynamics=True, project=True, u=Uin, tau=tau_pred, dt=dt, single_step=single_step,method=method)
#    Q_fut = pareo(Yin, tau=tau_pred, dt=dt)
#    
#    
#    if loss=='mae':
#        if prior is None:
#            w = None
#            def loss_wrapper(Y, Q_fut_hat, Q_fut, w):
#                def loss_f(y_true,y_pred):
#                    reconstruction = beta1*tf.reduce_mean(keras.losses.mean_absolute_error(Y,Xin[:,-1]))
#                    latent_pred = beta2*tf.reduce_mean(keras.losses.mean_absolute_error(Q_fut, Q_fut_hat))
#                    pred = beta3*tf.reduce_mean(keras.losses.mean_absolute_error(y_true,y_pred))
#                    return reconstruction + latent_pred + pred
#                return loss_f
#        else:
#            w = pareo.trainable_weights
#            if logspace and (not prior_logspace):
#                w = tf.exp(w)
#            if (not logspace) and prior_logspace:
#                w = tf.math.log(w)
#            if prior_logspace:
#                prior = tf.math.log(prior)
#            def loss_wrapper(Y, Q_fut_hat, Q_fut, w):
#                def loss_f(y_true,y_pred):
#                    reconstruction = beta1*tf.reduce_mean(keras.losses.mean_absolute_error(Y,Xin[:,-1]))
#                    latent_pred = beta2*tf.reduce_mean(keras.losses.mean_absolute_error(Q_fut, Q_fut_hat))
#                    pred = beta3*tf.reduce_mean(keras.losses.mean_absolute_error(y_true,y_pred))
#                    if prior_regularizer=='L1':
#                        prior_loss = beta_prior*tf.reduce_sum(tf.abs(w-prior))
#                    elif prior_regularizer=='L2':
#                        prior_loss = beta_prior*tf.reduce_sum(tf.square(w-prior))
#                    else:
#                        prior_loss = beta_prior*prior_regularizer(w,prior)
#                    return reconstruction + latent_pred + pred + prior_loss
#                return loss_f
#            
#    elif loss=='mse':
#        if prior is None:
#            w = None
#            def loss_wrapper(Y, Q_fut_hat, Q_fut, w):
#                def loss_f(y_true,y_pred):
#                    reconstruction = beta1*tf.reduce_mean(keras.losses.mean_squared_error(Y,Xin[:,-1]))
#                    latent_pred = beta2*tf.reduce_mean(keras.losses.mean_squared_error(Q_fut, Q_fut_hat))
#                    pred = beta3*tf.reduce_mean(keras.losses.mean_squared_error(y_true,y_pred))
#                    return reconstruction + latent_pred + pred
#                return loss_f
#        else:
#            w = pareo.trainable_weights
#            if logspace and (not prior_logspace):
#                w = tf.exp(w)
#            if (not logspace) and prior_logspace:
#                w = tf.math.log(w)
#            if prior_logspace:
#                prior = tf.math.log(prior)
#            def loss_wrapper(Y, Q_fut_hat, Q_fut, w):
#                def loss_f(y_true,y_pred):
#                    reconstruction = beta1*tf.reduce_mean(keras.losses.mean_squared_error(Y,Xin[:,-1]))
#                    latent_pred = beta2*tf.reduce_mean(keras.losses.mean_squared_error(Q_fut, Q_fut_hat))
#                    pred = beta3*tf.reduce_mean(keras.losses.mean_squared_error(y_true,y_pred))
#                    if prior_regularizer=='L1':
#                        prior_loss = beta_prior*tf.reduce_sum(tf.abs(w-prior))
#                    elif prior_regularizer=='L2':
#                        prior_loss = beta_prior*tf.reduce_sum(tf.square(w-prior))
#                    else:
#                        prior_loss = beta_prior*prior_regularizer(w,prior)
#                    return reconstruction + latent_pred + pred + prior_loss
#                return loss_f
#    
##    if loss=='mae':
##        def reconstruction_loss(Y):
##            return tf.reduce_mean(keras.losses.mean_absolute_error(Y,Xin[:,-1]))
##        def latent_pred_loss(Q_fut, Q_fut_hat):
##            return tf.reduce_mean(keras.losses.mean_absolute_error(Q_fut, Q_fut_hat))
##        def pred_loss(y_true, y_pred):
##            return tf.reduce_mean(keras.losses.mean_absolute_error(y_true,y_pred))
##    elif loss==
#
#
#    
#    model = keras.models.Model([Xin,Yin,Uin], Y_fut_hat)
#    if optimizer is None:
#        optimizer = keras.optimizers.Adam(learning_rate=lr)
#    model.compile(optimizer=optimizer,loss=loss_wrapper(Y_t,Q_fut_hat,Q_fut,w), ) 
#    par_layers = {'Q': Q,
#              'Y_t': Y_t,
#              'Q_fut_hat': Q_fut_hat,
#              'Y_fut_hat': Y_fut_hat,
#              'Q_fut': Q_fut,
#              'Xin': Xin,
#              'Yin': Yin,
#              'Uin': Uin,
#              'ParEO':  pareo}
#    return model, par_layers
#    
#def val_split(fract, batch_size, n):
#    return fract*n//batch_size*batch_size/n
#    
#def arctan_relative_loss(y_true, y_pred):
#    e = y_pred / y_true
#    L = K.abs(2*tf.math.atan(e)-tf.constant(np.pi/2))
##    L = np.abs(2*np.arctan(e)-)
#    return tf.reduce_mean(L, axis = -1)
#
#def fitted_inferenceModel(par_layers):
#    model = keras.models.Model(par_layers['Xin'],par_layers['Q'])
#    return model
    