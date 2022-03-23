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
dt=1/100

beta =8.0/3
rho= 28.0
sigma= 10.0
latent_dim = 3
#DEFINE EMBEDDING PARAMETERS FOR DATA  
m = 50
tau_lag = 3
tau_pred = 100#30
#generate data
def lorenz(X,t,beta, rho, sigma):
    x = X[0]
    y = X[1]
    z = X[2]
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return [xdot, ydot, zdot]
const = (beta, rho,sigma)
from scipy.integrate import odeint
t = np.arange(0,2000,dt)
data_full = []
XX = []
YY = []
UU = []
for i in range(1):
    init=[1,1,1]
    data=odeint(lorenz,init,t,const)
    #data += np.random.normal(0,.1,data.shape)
    data_full.extend(data[:data.shape[0]-m*tau_lag+1-tau_pred])
    data = np.array([data[i:i+m*tau_lag:tau_lag,0] for i in range(data.shape[0]-m*tau_lag+1)])
    XX.extend(data[:-tau_pred])
    YY.extend(data[tau_pred:])
    UU.extend(0 * np.ones((data.shape[0]-tau_pred,tau_pred)))
       
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
from ParEO.measurement import M_project
M = M_project(0)

# In[]
"""
Build ParEO Network
"""
from ParEO.ParEO import ParEO_network
from ParEO.dynamics import lorenz as theta
batch_size = 64
parameter_logspace = 0
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
epochs=1000
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
"""
attractor
"""
from mpl_toolkits.mplot3d import Axes3D
mx=10000
fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132,)
ax3 = fig.add_subplot(133, projection='3d')
ax.plot(data_full[m*tau_lag:mx,0],data_full[m*tau_lag:mx,1],data_full[m*tau_lag:mx,2],lw=.5)
ax3.plot(latent_hat[:mx,0],latent_hat[:mx,1],latent_hat[:mx,2],lw=.5)

ax2.plot(data_full[m*tau_lag:mx,0],)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_xlim(0,10000)
ax.set_title('true data')
ax2.set_title('observed')
ax3.set_title('learned reconstruction')

# In[]
sh=m*tau_lag
plt.scatter(XX[:,-1],latent_hat[:,0],s=10)


# In[]
"""
prediction error
"""
y_hat = pareo.predict([XX,YY,UU],batch_size=4096)

#plt.plot(XX[:,-1])
#plt.plot(YY[:,-1])
#plt.plot(y_hat)
plt.scatter(YY[:,-1],y_hat,s=10,alpha=.1)

# In[]
"""
Prediction Error vs. tau_pred
"""
latent_ = N.predict(XX,batch_size=2048)
l0 = latent_[:,0].copy()
w = lay['THETA'].get_weights()
if parameter_logspace: w = np.exp(w)
beta,rho,sigma = w[0]

xx = []
error = []
error_auto = []
x = latent_[:,0]
y = latent_[:,1]
z = latent_[:,2]
for i in range(1,1000):
#    dx = sigma * (y - x)
#    dy = x * (rho - z) - y
#    dz = x * y - beta * z
#    x += dx*dt
#    y += dy*dt
#    z += dz+dt
    x += (sigma * (y - x))*dt
    y += (x * (rho - z) - y)*dt
    z += (x * y - beta * z)*dt
    xx.append(latent_[1000,0])
    error.append((x[:-i]-XX[i:,-1])**2)
    error_auto.append((l0[:-i]-XX[i:,-1])**2)
    
plt.plot(xx)
plt.plot(XX[1000:,-1])
# In[]
##pareo error
e_ = [np.mean(e) for e in error]
#e_ = [np.median(e) for e in error]
#e_lo = [np.percentile(e,25) for e in error]
#e_hi = [np.percentile(e,75) for e in error]
#autoregressive-0 error
ea_ = [np.mean(e) for e in error_auto]
#ea_lo = [np.percentile(e,25) for e in error_auto]
#ea_hi = [np.percentile(e,75) for e in error_auto]
#ea_lo = [np.mean(e)-np.std(e) for e in error_auto]
#ea_hi = [np.mean(e)+np.std(e) for e in error_auto]

tt = np.arange(len(e_))*dt
#tt = np.log10(tt)
#e_ = np.log10(e_)
#e_lo = np.log10(e_lo)
#e_hi = np.log10(e_hi)

fig=plt.figure()
ax=fig.gca()
plt.plot(tt,e_,label='pareo model')
#plt.fill_between(tt,e_lo,e_hi,alpha=.1)
plt.plot(tt,ea_,label='A0 model')
#plt.fill_between(tt,ea_lo,ea_hi,alpha=.1)
lyap = 1.0910931847726466 #see dyna database
plt.plot([lyap,lyap],[-10,150,],c='grey',ls=':',zorder=-1,label='$\lambda_{lyapanov}$')
plt.plot([tau_pred*dt,tau_pred*dt],[-10,150,],c='navy',ls='--',zorder=-1,label='$\\tau_{predict}$')

plt.ylabel('$\langle MSE \\rangle$')
plt.xlabel('prediction time')
plt.legend()
plt.xlim(-.1,tt.max())
plt.ylim(-2,130)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

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
aa = np.linspace(-.5,1,10)
bb = np.linspace(-.5,1,10)
loss_ = np.zeros((aa.size,bb.size))
for i in tqdm(range(aa.size)):
    for j in range(bb.size):
        lay['THETA'].set_weights([np.array([aa[i],bb[j],8])])
        e = pareo.evaluate(x=[XX,YY,UU], y=Y_out,batch_size=4096,verbose=0)
        loss_[i,j] = e
    
# In[]
plt.imshow((loss_),origin='lower',aspect='auto',extent=(aa[0],aa[-1],bb[0],bb[-1]))
    
plt.colorbar() 
    
    
    
    
    