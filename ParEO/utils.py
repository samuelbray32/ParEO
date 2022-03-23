#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:35:43 2021

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
#from analysisScripts.latentModel_tf.data_prep import latentModel_dataPrep_fromRecord, latentModel_dataPrep_generic

def prepare_data(data):
    XX = []
    YY = []
    UU = []
    tau = result['tau']
    for i,pulse in enumerate(tags):
        yp = result[data+pulse] + shift_data
        print(data+pulse,yp.shape)
        u = np.zeros(yp.shape[1])
        l = np.argmin(tau**2)
        u[l:l+sh[i]] = 1
        ind = np.arange(yp.shape[0])
        np.random.shuffle(ind)
        while ind.size<max_pulse:
            ind = np.append(ind,ind)
        ind=ind[:max_pulse]
        for y in yp[ind]:
            z, z_fut, u_ = latentModel_dataPrep_fromRecord(y,u,m,tau_lag,tau_pred,light_sample, embed_stimulus=embed_stimulus, return_U=True) 
        #    u_ = np.moveaxis(np.array([u_,1-u_,]),0,2)
#            if log_data:
#                z[z<0]=.001
#                z_fut[z_fut<0]=.001
#                z=np.log10(z)
#                z_fut=np.log10(z_fut)
            UU.extend(u_)
            XX.extend(z)
            YY.extend(z_fut)
    XX = np.array(XX)
    YY = np.array(YY)
    UU = np.array(UU)
    return XX,YY,UU

