#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:03:46 2021

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
config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()

def stimulus(period):
    if period==0:
        return lambda x: 0*x+1
    return lambda x: 1+1*np.sin(x/period)

def eval_fit(a,b,tau,
             epochs=10,train_log=True):
    """
    GENERATE TRAINING DATA
    """
    #define  model parameters
    dt=1/10
    latent_dim = 2
    #DEFINE EMBEDDING PARAMETERS FOR DATA  
    m = 20
    tau_lag = 5
    tau_pred = 20
    #generate data
    def fitz_nagumo(x,t,a,b,tau,I):
        v = x[0]
        w = x[1]
        dv = v - v**3/3 - w +I(t)
        dw = (v +a -b*w)/tau
        return [dv,dw]
    from scipy.integrate import odeint
    t = np.arange(0,500,dt)
    data_full = []
    XX = []
    YY = []
    UU = []
    for i in range(10):
        u = stimulus(i*10)
        const = (a,b,tau,u)
        init=[0,0]#np.random.uniform(-5,5,2)
        data=odeint(fitz_nagumo,init,t,const)
        data += np.random.normal(0,.5,data.shape)
        U = u(t)   
        data_full.extend(data[:data.shape[0]-m*tau_lag+1-tau_pred])
        data = np.array([data[i:i+m*tau_lag:tau_lag,0] for i in range(data.shape[0]-m*tau_lag+1)])
        U_emb = np.array([U[i:i+m*tau_lag:tau_lag] for i in range(U.shape[0]-m*tau_lag+1)])
        data = np.append(U_emb,data,axis=1)
        U_stim = np.array([U[i+m*tau_lag:i+m*tau_lag+tau_pred] for i in range(U.shape[0]-m*tau_lag+1)])
        XX.extend(data[:-tau_pred])
        YY.extend(data[tau_pred:])
        UU.extend(U_stim[:-tau_pred])
           
    data_full = np.array(data_full)
    XX = np.array(XX)
    YY = np.array(YY)
    UU = np.array(UU)
    
    #In[]
    """
    Define Embedding and Measurement Models/Layers
    """
    from ParEO.embedding import inferenceNetwork_RNN
    N = inferenceNetwork_RNN(context_dim=XX.shape[1], parameter_dim=None, 
                         postRNN_layers=[], parameter_layers=[],
                         latent=latent_dim, activation='relu',regular=None,
                         dropout=0,anchor_ind=-1,
                         square_output=False, stimulus=True, conditioning=False)
    from ParEO.measurement import M_project
    M = M_project(0)
    
    #In[]
    """
    Build ParEO Network
    """
    from ParEO.ParEO import ParEO_network
    from ParEO.dynamics import fitz_nagumo as theta
    batch_size = 32
    parameter_logspace = train_log
    lr = 2e-3#3e-3#1e-2
    initial='ones'
    if parameter_logspace: initial='zeros'
    beta1= 1e0#3e2
    beta2= 3e1#5e3
    beta3= 3e1#2e3
    beta_sum = beta1+beta2+beta3
    beta1 /= beta_sum
    beta2 /= beta_sum
    beta3 /= beta_sum
    #beta1=1e0, beta2=1e1, beta3=3e1
    pareo,lay = ParEO_network(N, theta, batch_size, tau_pred, dt, XX.shape[1:],
                      M=M, initializer=initial,
                      lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, input_d=1, method='taylor',
                      logspace=parameter_logspace, loss='mse', single_step=False, optimizer=None)
    
    #In[]
    """
    Train Complete Model
    """
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
                  batch_size=batch_size,epochs=epochs, verbose=1,shuffle=True,
                  )
    return lay['THETA'].get_weights()[0]

# In[]
train_log=False
epochs=30
p = []
p_hat = []

for i in range(3000):
    p_i = np.zeros(3)
    p_i[:2] = np.random.uniform(0,2,2)
    p_i[2] = np.random.uniform(0,10,1)#10**np.random.uniform(-.5,1,1)
    p_hat_i = eval_fit(*p_i,epochs=epochs,train_log=train_log)
    if train_log:
        p_hat_i = np.exp(p_hat_i)
    print(f'ROUND {i}')
    print(p_i,)
    print(p_hat_i)
    p.append(p_i)
    p_hat.append(p_hat_i)
           
        
# In[]
''' seperate reults'''
from scipy.stats import linregress
from ParEO.dynamics import fitz_nagumo as theta
log = False
fig, ax = plt.subplots(ncols=p[0].shape[0])
for i,a in enumerate(ax):
    xx = np.array(p)[:,i].copy()
    yy = np.array(p_hat)[:,i].copy()
    if log:
        xx=np.log10(xx)
        yy=np.log10(yy)
    ind = np.where(np.isfinite(yy))[0]
    q = linregress(xx[ind],yy[ind])
    
    a.scatter(xx,yy,label=f'r={np.round(q[2],3)}')
    a.plot(xx,xx,ls=':',c='k',zorder=-1)
    a.set_title(theta.param_names()[i])
    a.legend()
xlab='p'
ylab='$\hat p$'
if log:
    xlab='log10 '+xlab
    ylab='log10 '+ylab
ax[len(ax)//2].set_xlabel(xlab)
ax[0].set_ylabel(ylab)

# In[]
'''error correlation'''
from scipy.stats import linregress
from ParEO.dynamics import fitz_nagumo as theta
log = False
fig, ax = plt.subplots(ncols=p.shape[1])
for i,a in enumerate(ax):
    xx = p_hat[:,i]-p[:,i]
    yy = p_hat[:,i-1]-p[:,i-1]
    if log:
        xx=np.log10(p_hat[:,i]/p[:,i])
        yy=np.log10(p_hat[:,i-1]/p[:,i-1])
    q = linregress(xx,yy)
    a.scatter(xx,yy,label=f'r={np.round(q[2],3)}')
    a.set_xlabel(theta.param_names()[i])
    a.set_ylabel(theta.param_names()[i-1])
    a.legend()

