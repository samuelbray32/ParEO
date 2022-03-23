#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:47:24 2021

@author: sam
"""

# In[]
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

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
#config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()

def eval_fit(k):
#    #define  model parameters
#    dt=1/100
#    k = 3
#    latent_dim = 2
#    #generate data
#    def spring (x,t,k):
#        return [x[1],-k*x[0]]
#    from scipy.integrate import odeint
#    t = np.arange(0,100,dt)
#    data_full = []
#    for i in range(30):
#        init=np.random.uniform(0,1,2)
#        data_full.extend(odeint(spring,init,t,(k,)))
#    data_full = np.array(data_full)
#
#    #DEFINE EMBEDDING PARAMETERS FOR DATA
#    m = 20
#    tau_lag = 3
#    tau_pred = 50  
#    data = np.array([data_full[i:i+m*tau_lag:tau_lag,0] for i in range(data_full.shape[0]-m*tau_lag+1)])
#    XX = data[:-tau_pred]
#    YY = data[tau_pred:]
#    UU = np.zeros((XX.shape[0],tau_pred))
    #define  model parameters
    dt=1/100
    latent_dim = 2
    #DEFINE EMBEDDING PARAMETERS FOR DATA  
    m = 20
    tau_lag = 3
    tau_pred = 50
    #generate data
    def spring (x,t,k):
        return [x[1],-k*x[0]]
    from scipy.integrate import odeint
    t = np.arange(0,100,dt)
    data_full = []
    XX = []
    YY = []
    UU = []
    for i in range(30):
        init=np.random.uniform(0,3,2)
        data=odeint(spring,init,t,(k,))
        data_full.extend(data[:data.shape[0]-m*tau_lag+1-tau_pred])
        data = np.array([data[i:i+m*tau_lag:tau_lag,0] for i in range(data.shape[0]-m*tau_lag+1)])
        XX.extend(data[:-tau_pred])
        YY.extend(data[tau_pred:])
        UU.extend(np.zeros((data.shape[0]-tau_pred,tau_pred)))
           
    data_full = np.array(data_full)
    XX = np.array(XX)
    YY = np.array(YY)
    UU = np.array(UU)

    #Define Embedding and Measurement Models/Layers
    from ParEO.embedding import inferenceNetwork_RNN
    N = inferenceNetwork_RNN(context_dim=m, parameter_dim=None, 
                         postRNN_layers=[], parameter_layers=[],
                         latent=latent_dim, activation='relu',regular=None,
                         dropout=0,anchor_ind=-1,
                         square_output=False, stimulus=False, conditioning=False)
    from ParEO.measurement import M_project
    M = M_project(0)

    # Build ParEO Network
    from ParEO.ParEO import ParEO_network
    from ParEO.dynamics import spring as theta
    batch_size = 256
    parameter_logspace = False
    lr = 1e-2
    initial='ones'
    rho = .999
    beta1= 1e0#3e2
    beta2= 3e1#5e3
    beta3= 3e1#2e3
    beta_sum = beta1+beta2+beta3
    beta1 /= beta_sum
    beta2 /= beta_sum
    beta3 /= beta_sum    
    pareo,lay = ParEO_network(N, theta, batch_size, tau_pred, dt, XX.shape[1:],
                      M=M, initializer='ones',
                      lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, input_d=1, method='taylor',
                      logspace=False, loss='mse', single_step=False, optimizer=None)

    #Train Complete Model
    epochs=10
    ind=None#np.arange(0,10**4)
    if ind is None:
        Y_out = np.squeeze(YY[:,-1])
        Y_out = np.append(Y_out[:,None],Y_out[:,None],axis=-1)
        pareo.fit(x=[XX,YY,UU], y=Y_out,
                  batch_size=batch_size,epochs=epochs, verbose=1,shuffle=True,
                  )
    else:
        Y_out = YY[ind,-1]
        Y_out = np.append(Y_out[:,None],Y_out[:,None],axis=-1)
        pareo.fit(x=[XX[ind],YY[ind],UU[ind]], y=Y_out,
                  batch_size=batch_size,epochs=epochs, verbose=1,
                  )
    return lay['THETA'].get_weights()[0]

# In[]
k = np.logspace(-1,1,10)
k_hat = np.zeros_like(k)
for i in range(k.size):
    k_hat[i] = eval_fit(k[i])
    print(k[i],k_hat[i])

# In[]
from scipy.stats import linregress
log = False
xx = k.copy()
yy = k_hat.copy()
if log:
    xx=np.log10(xx)
    yy=np.log10(yy)
q = linregress(xx,yy)
plt.scatter(xx,yy,label=f'r={np.round(q[2],3)}')
plt.plot(xx,xx,ls=':',c='k',zorder=-1)

plt.legend()
xlab='k'
ylab='$\hat k$'
if log:
    xlab='log10 '+xlab
    ylab='log10 '+ylab
plt.xlabel(xlab)
plt.ylabel(ylab)










