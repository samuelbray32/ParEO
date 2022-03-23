#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 08:56:46 2021

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

class theta_model1_1():
    def dynamics(X, P, u, tau, dt, single_step=False):
        #unpack parameters
        mu=P[0]
        phi=P[1]
        gamma=P[2]
        delta=P[3]
        k=P[4]
        eta=P[5]
        
        a = X[:,0]
        i = X[:,1]
        if single_step:
            a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (phi * i) + eta * tf.reduce_mean(u,axis=1))*dt
            i = i+((gamma * a) - (delta * i))*dt
        else:
            for j in range(u.get_shape().as_list()[1]):
                a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (phi * i) + eta * u[:,j])*dt
                i = i+((gamma * a) - (delta * i))*dt
        a = tf.expand_dims(a,-1)
        i = tf.expand_dims(i,-1)
        return tf.concat([a,i], axis=1)
    
class theta_model1_2():
    def dynamics(X, P, u, tau, dt, single_step=False, method='taylor'):
        #unpack parameters
        mu=P[0]
        gamma=P[1]
        delta=P[2]
        k=P[3]
        eta=P[4]
        
        a = X[:,0]
        i = X[:,1]
#        if single_step:
#            a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (1 * i) + eta * tf.reduce_mean(u,axis=1))*dt
#            i = i+((gamma * a) - (delta * i))*dt
#        else:
#            for j in range(u.get_shape().as_list()[1]):
#                a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (1 * i) + eta * u[:,j])*dt
#                i = i+((gamma * a) - (delta * i))*dt
        if method=='taylor':
            if single_step:
                a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (1 * i) + eta * tf.reduce_mean(u,axis=1))*dt*u.get_shape().as_list()[1]
                i = i+((gamma * a) - (delta * i))*dt*u.get_shape().as_list()[1]
            else:
                for j in range(u.get_shape().as_list()[1]):
                    a =  a+((mu * tf.square(a)) / (k + tf.square(a)) - (1 * i) + eta * u[:,j])*dt
                    i = i+((gamma * a) - (delta * i))*dt
        elif method=='RK':
            def grad(a,i,u):
                da = (mu * tf.square(a)) / (k + tf.square(a)) - (1 * i) + eta * u
                di = (gamma * a) - (delta * i)
                return da,di
            def rk(a,i,u,h):
                k1a, k1i = grad(a,i,u)
                k2a, k2i = grad(a+k1a/2*h,i+k1i/2*h,u)
                k3a, k3i = grad(a+k2a/2*h,i+k2i/2*h,u)
                k4a, k4i = grad(a+k3a*h,i+k3i*h,u)
                return a+h/6*(k1a+2*k2a+2*k3a+k4a), i+h/6*(k1i+2*k2i+2*k3i+k4i)
            
            if single_step:
                a, i = rk(a,i,tf.reduce_mean(u,axis=1),dt*u.get_shape().as_list()[1])    
            else:
               for j in range(u.get_shape().as_list()[1]):
                   a, i = rk(a,i,u[:,j],dt)
        
        a = tf.expand_dims(a,-1)
        i = tf.expand_dims(i,-1)
        return tf.concat([a,i], axis=1)

    def param_names():
        return ['mu','gamma','delta','k','eta']
    
    
class theta_model3():
    def dynamics(X, P, u, tau, dt):
        #unpack parameters
        mu1 = P[0]
        mu2 = P[1]
        delta1 = P[2]
        delta2 = P[3]
        gamma1 = P[4]
        gamma2 = P[5]
        
        a = X[:,0]
        s = X[:,1]
        for j in range(u.get_shape().as_list()[1]):
            a = a + ((mu1*(1-a) - delta1*s)*a + gamma1*u[:,j,0])/120
            s = s + ((mu2*(1-s) - delta2*a)*s + gamma2*u[:,j,1])/120
        a = tf.expand_dims(a,-1)
        s = tf.expand_dims(s,-1)
        return tf.concat([a,s], axis=1)
    def param_names():
        return ['mu1','mu2','delta1','delta2','gamma1','gamma2']
    

class theta_model_neurons():
    #Models latent dynamics as recurrently connected neurons and firing rates
    #Based on "Training Excitatory-Inhibitory RecurrentNeural Networks for Cognitive Tasks: ASimple and Flexible Framework"
    #Song, Yang, Wang, 2016
    def __init__(self,n=100,n_stim=1,connectivity=1e-1,excitatory=8e-1):
        #define non-trainable components
        self.n = n
        self.n_stim = n_stim
        #neural connectivity matrix
        if type(connectivity) is float:
            assert (connectivity>=0) and (connectivity<=1)
            connectivity = np.random.uniform(0,1,(n,n))<=connectivity
        self.connectivity = tf.constant(connectivity,dtype='float32')
        #excitatory/inhibitory diagonal matrix
        if type(excitatory) is float:
            assert (excitatory>=0) and (excitatory<=1)
            temp = (np.random.uniform(0,1,n)<=excitatory)*2-1
            excitatory = np.zeros((n,n))
            for i in range(n):
                excitatory[i,i] = temp[i]
        self.excitatory = tf.constant(excitatory,dtype='float32')
                   
    def dynamics(self,X, P, u, dt):
        #unpack parameters
        #rectify W to positive values
        W = tf.nn.relu(tf.reshape(P[:self.n**2],(self.n,self.n)))
        #apply excitatory/inhibitory constraint
        W =  K.dot(self.excitatory,W,)#tf.matmul(W,self.excitatory,b_is_sparse=True)
        #apply connectivity constraint
        W = tf.math.multiply(W,self.connectivity)
        #stimulus weight matrix
        W_in = tf.reshape(P[self.n**2:],(self.n_stim,self.n))
        #run dynamics
        print(W.shape,W_in.shape,)
        for j in range(u.get_shape().as_list()[1]):
            #rectify state for firing rates
            r = tf.nn.relu(X)
            #dynamic update of state
            X = X + (-X + K.dot(r,W)) + K.dot(u[:,j,:],W_in)*dt
        return X
    
    def param_names(self,):
        names=[]
        for i in range(self.n):
            for j in range(self.n):
                names.append('W_'+str(i)+'_'+str(j))
        for i in range(self.n):
            for j in range(self.n_stim):
                names.append('Win_'+str(i)+'_'+str(j)) 
        return names
    
class spring():
    def dynamics(X, P, u, dt):
        #unpack parameters
        k = P[0]
        #turn xy state into radius and angle
        x = X[:,0]
        v = X[:,1]
        #prpogate dynamics
        for j in range(u.get_shape().as_list()[1]):
            x  = x + (v+u[:,j])*dt
            v = v + (-k*x)*dt 
        #convert back to xy and return
        x = tf.expand_dims(x,-1)
        v = tf.expand_dims(v,-1)
        return tf.concat([x,v], axis=1)
    def param_names():
        return ['k']

class lorenz():
    def dynamics(X,P,u,dt):
        #unpack parameters
        beta = P[0]
        rho = P[1]
        sigma = P[2]
        #turn xy state into radius and angle
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]
        #prpogate dynamics
        for j in range(u.get_shape().as_list()[1]):
            x = x + (sigma * (y - x))*dt
            y = y + (x * (rho - z) - y)*dt
            z = z + (x * y - beta * z)*dt
        #convert back to xy and return
        x = tf.expand_dims(x,-1)
        y = tf.expand_dims(y,-1)
        z = tf.expand_dims(z,-1)
        return tf.concat([x,y,z], axis=1)
    def param_names():
        return ['beta','rho','omega']
   
class fitz_nagumo():
    def dynamics(X, P, u, dt):
        #unpack parameters
        a = P[0]
        b = P[1]
        tau = P[2]
        #I = P[3]
        #turn xy state into radius and angle
        v = X[:,0]
        w = X[:,1]
        #prpogate dynamics
        for j in range(u.get_shape().as_list()[1]):
            v  = v + (v - v**3/3 - w + u[:,j])*dt
            w = w + ((v +a -b*w)/tau)*dt 
        #convert back to xy and return
        w = tf.expand_dims(w,-1)
        v = tf.expand_dims(v,-1)
        return tf.concat([v,w], axis=1)
    def param_names():
        return ['a','b','tau',]#'I']

class fitz_nagumo_v2():
    '''See: Parameter Estimation with Dense and Convolutional NeuralNetworks Applied to the FitzHugh–Nagumo ODE'''
    def dynamics(X, P, u, dt):
        #unpack parameters
        a = P[0]
        b = P[1]
        #locked parameter
        tau = 3.0
        #turn xy state into radius and angle
        v = X[:,0]
        w = X[:,1]
        #propagate dynamics        
        for j in range(u.get_shape().as_list()[1]):
            v  = v + (v - v**3/3 + w + u[:,j])*tau*dt
            w = w + ((v -a +b*w)/tau)*dt 
        #convert back to xy and return
        w = tf.expand_dims(w,-1)
        v = tf.expand_dims(v,-1)
        return tf.concat([v,w], axis=1)
    def param_names():
        return ['a','b',]#'I']