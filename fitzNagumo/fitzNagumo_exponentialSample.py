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



# In[]
"""
GENERATE TRAINING DATA
"""
#define  model parameters
dt=1/10
a = .7
b=.8
tau=8
I=1
latent_dim = 2
#DEFINE EMBEDDING PARAMETERS FOR DATA  
m = 33
tau_lag = 3
tau_pred = 20
#generate data
def fitz_nagumo(x,t,a,b,tau,I):
    v = x[0]
    w = x[1]
    dv = v - v**3/3 - w +I
    dw = (v +a -b*w)/tau
    return [dv,dw]
def noise_model(x):
    return np.random.exponential(np.exp(x))

const = (a,b,tau,I)
from scipy.integrate import odeint
t = np.arange(0,100,dt)
data_full = []
XX = []
YY = []
UU = []
for i in range(100):
    init=[0,0]#np.random.uniform(-5,5,2)
    data=odeint(fitz_nagumo,init,t,const)
    data_full.extend(data[:data.shape[0]-m*tau_lag+1-tau_pred])
    data = np.array([data[i:i+m*tau_lag:tau_lag,0] for i in range(data.shape[0]-m*tau_lag+1)])
    data = noise_model(data)
    XX.extend(data[:-tau_pred])
    YY.extend(data[tau_pred:])
    UU.extend(I * np.ones((data.shape[0]-tau_pred,tau_pred)))
       
data_full = np.array(data_full)
XX = np.array(XX)
YY = np.array(YY)
UU = np.array(UU)
   
plt.plot(data[:,0])

# In[]
"""
Define Embedding and Measurement Models/Layers
"""
from ParEO.embedding import inferenceNetwork_RNN
N = inferenceNetwork_RNN(context_dim=m, parameter_dim=None, 
                     postRNN_layers=[], parameter_layers=[],
                     latent=latent_dim, activation='relu',regular=None,
                     dropout=0,anchor_ind=-1,
                     square_output=False, stimulus=False, conditioning=False)
from ParEO.measurement import M_project_exponential
M = M_project_exponential(0)

# In[]
"""
Build ParEO Network
"""
from ParEO.ParEO import ParEO_network
from ParEO.dynamics import fitz_nagumo as theta
batch_size = 128
parameter_logspace = False
lr = 1e-3
initial='ones'
if parameter_logspace: initial='zeros'
rho = .999
beta1= 1e1#3e2
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
                  logspace=parameter_logspace, loss='exponential', single_step=False, optimizer=None)

# In[]
"""
Train Complete Model
"""
epochs=10000
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

# In[]
plt.plot(pareo.history.history['loss'])
w = lay['THETA'].get_weights()
if parameter_logspace: w = np.exp(w)
print(w)
print(const)
# In[]
"""
reconstruction error
"""
mx = 2048
latent_hat = N.predict(XX[:mx],batch_size=2048)
fig,ax = plt.subplots(nrows=2,sharex=True,sharey=False)
ax[0].plot(data_full[m*tau_lag:][:mx])
ax[1].plot(latent_hat)
ax[0].set_title('true')
ax[1].set_title('predicted')

fig,ax = plt.subplots(nrows=latent_dim,sharex=True)
for i,a in enumerate(ax):
    a.plot(data_full[m*tau_lag:,i][:mx])
    a.plot(latent_hat[:,i])
    a.set_title(f'latent dim {i}')

# In[]
sh=m*tau_lag
plt.scatter(XX[:,-1],latent_hat[:,0],s=10)


# In[]
"""
prediction error
"""
y_hat = pareo.predict([XX[:mx],YY[:mx],UU[:mx]])

plt.plot(YY[:,-1])
plt.plot(y_hat)
# In[]

latent_pred = keras.models.Model(pareo.layers[0].input,pareo.layers[2].output)
q_fut_hat = latent_pred.predict(XX,batch_size=2048)
q_fut = N.predict(YY,batch_size=2048)

fig,ax = plt.subplots(nrows=latent_dim,sharex=True)
for i,a in enumerate(ax):
    a.plot(q_fut_hat[:,i])
    a.plot(q_fut[:,i])
    a.set_title(f'latent dim {i}')
