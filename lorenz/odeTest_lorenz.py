#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 12:15:51 2021

@author: sam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# In[]

def lorenz(X,t,sigma, rho, beta):
    x = X[0]
    y = X[1]
    z = X[2]
    xdot = sigma * (y - x)
    ydot = x * (rho - z) - y
    zdot = x * y - beta * z
    return [xdot, ydot, zdot]

#beta =8.0/3
#rho= 28.0
#sigma= 10.0
dt = 1/100
init=[1,1,1]
const=(sigma,rho,beta)
t  = np.arange(0,400,dt)
X = odeint(lorenz,init,t,const)
plt.plot(t,X)
# In[]
X2=odeint(fitz_nagumo,init,t,const2)
fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(X)
ax[1].plot(X2)
