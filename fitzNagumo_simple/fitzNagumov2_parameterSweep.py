#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:17:22 2021

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

MEASURE_INDEX=0

def eval_fit(a,b,tau=3,I=-.4,
             epochs=30,train_log=False,init_p=None):
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
    dt=.1
    latent_dim = 2
    #DEFINE EMBEDDING PARAMETERS FOR DATA  
    m = 20
    tau_lag = 5#5
    tau_pred = 50#20
    #generate data
    def fitz_nagumo(x,t,a,b,tau,I):
        v = x[0]
        w = x[1]
        dv = (v - v**3/3 + w + I)*tau
        dw = -(v -a +b*w)/tau
        return [dv,dw]
    const = (a,b,tau,I)
    from scipy.integrate import odeint
    t = np.arange(0,100,dt)
    data_full = []
    XX = []
    YY = []
    UU = []
    for i in range(50):
        init=np.random.uniform(-5,5,2)
        data=odeint(fitz_nagumo,init,t,const)
        #data += np.random.normal(0,.1,data.shape)
        data_full.extend(data[:data.shape[0]-m*tau_lag+1-tau_pred])
        data = np.array([data[i:i+m*tau_lag:tau_lag,MEASURE_INDEX] for i in range(data.shape[0]-m*tau_lag+1)])
        XX.extend(data[:-tau_pred])
        YY.extend(data[tau_pred:])
        UU.extend(I * np.ones((data.shape[0]-tau_pred,tau_pred)))
           
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
    M = M_project(MEASURE_INDEX)

    # Build ParEO Network
    from ParEO.ParEO import ParEO_network
    from ParEO.dynamics import fitz_nagumo_v2 as theta
    batch_size = 32
    lr = 3e-3
    initial='ones'
    if train_log: initial='zeros'
    beta1= 1e0#3e2
    beta2= 3e1#5e3
    beta3= 3e1#2e3
    beta_sum = beta1+beta2+beta3
    beta1 /= beta_sum
    beta2 /= beta_sum
    beta3 /= beta_sum    
    pareo,lay = ParEO_network(N, theta, batch_size, tau_pred, dt, XX.shape[1:],
                      M=M, initializer=initial,
                      lr=lr, beta1=beta1, beta2=beta2, beta3=beta3, input_d=1, method='taylor',
                      logspace=train_log, loss='mse', single_step=False, optimizer=None)

    if not init_p is None:
        lay['THETA'].set_weights([init_p])
        print(init_p)
    #Train Complete Model
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
train_log=True
epochs=10
p = np.random.uniform(0,1,(200,2))
#p[:,2] = 10**np.random.uniform(-.5,1,p.shape[0])
p_hat = np.zeros_like(p)

for i in range(p.shape[0]):
    print(p[i])
    p_hat[i] = eval_fit(*p[i],epochs=epochs,train_log=train_log,)
    print(f'ROUND {i}')
    print(p[i],)
    if train_log:
        print(np.exp(p_hat[i]))
    else:
        print(p_hat[i])
# In[]
p=p[:i]
p_hat=p_hat[:i]
if train_log:p_hat = np.exp(p_hat)
        
# In[]
''' seperate reults'''
from scipy.stats import linregress
from ParEO.dynamics import fitz_nagumo as theta
log = False
fig, ax = plt.subplots(ncols=p.shape[1])
for i,a in enumerate(ax):
    xx = p[:,i].copy()
    yy = p_hat[:,i].copy()
    if log:
        xx=np.log10(xx)
        yy=np.log10(yy)
    q = linregress(xx,yy)
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
log = True
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

