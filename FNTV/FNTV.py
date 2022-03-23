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
plt.rcParams['svg.fonttype'] = 'none'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
#config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()

def stimulus(period):
    if period==0:
        return lambda x: 0*x+1
    return lambda x: 1+1*np.sin(x/period)

# In[]
"""
GENERATE TRAINING DATA
"""
#define  model parameters
dt=1/10
#a = .7
#b=.8
#tau=8
a = 1.7
b=.8
tau=3
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
    plt.plot(data[:,0])
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
   
#plt.plot(data[:,:])

# In[]
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

# In[]
"""
Build ParEO Network
"""
from ParEO.ParEO import ParEO_network
from ParEO.dynamics import fitz_nagumo as theta
batch_size = 32
parameter_logspace = False
lr = 3e-3#1e-2
initial='ones'
if parameter_logspace: initial='zeros'
rho = .999
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

# In[]
"""
Train Complete Model
"""
epochs=100
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
latent_hat = N.predict(XX,batch_size=2048)
fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(data_full[m*tau_lag:])
ax[1].plot(latent_hat)
ax[0].set_title('true')
ax[1].set_title('predicted')

fig,ax = plt.subplots(nrows=latent_dim,sharex=True)
for i,a in enumerate(ax):
    a.plot(data_full[m*tau_lag:,i])
    a.plot(latent_hat[:,i])
    a.set_title(f'latent dim {i}')

# In[]
sh=m*tau_lag
plt.scatter(XX[:,-1],latent_hat[:,0],s=10)


# In[]
"""
prediction error
"""
y_hat = pareo.predict([XX,YY,UU])

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

# In[]
from tqdm import tqdm
aa = np.linspace(-.5,2.5,100)
bb = np.linspace(-.5,2.5,100)
loss_ = np.zeros((aa.size,bb.size))
p_fit = lay['THETA'].get_weights()
sub=3
for i in tqdm(range(aa.size)):
    for j in range(bb.size):
        lay['THETA'].set_weights([np.array([aa[i],bb[j],p_fit[0][-1]])])
        e = pareo.evaluate(x=[XX[::sub],YY[::sub],UU[::sub]], y=Y_out[::sub],batch_size=4096,verbose=0)
        loss_[i,j] = e
lay['THETA'].set_weights(p_fit)    
# In[]
plt.imshow(np.log10(loss_),origin='lower',aspect='auto',extent=(aa[0],aa[-1],bb[0],bb[-1]))
plt.colorbar(label='$log10\ \scrL$') 
plt.scatter([b,],[1.7,],c='r',marker='x',s=100,label='True Value')    
min_ = np.unravel_index(loss_.argmin(), loss_.shape)
plt.scatter([bb[min_[1]],],[aa[min_[0]],],c='g',marker='+',s=100,label='optimum')    
plt.xlabel('b')
plt.ylabel('a')
plt.legend()