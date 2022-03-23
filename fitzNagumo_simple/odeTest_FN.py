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

def fitz_nagumo(x,t,a,b,tau,I):
    v = x[0]
    w = x[1]
    dv = (v - v**3/3 + w + I)*tau
    dw = -(v -a +b*w)/tau
    return [dv,dw]

dt=.1

a = .001
b=.001
tau=3
I=-.4
init=[0,1]
const=(a,b,tau,I)
const2=(0.5561093 , 0.8056585 , 1.9398959 , 0.81629777 )
t  = np.arange(0,100,dt)
X = odeint(fitz_nagumo,init,t,const)
#X2=odeint(fitz_nagumo,init,t,const2)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(X)
#ax[1].plot(X2)
