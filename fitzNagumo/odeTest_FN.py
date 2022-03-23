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
    dv = v - v**3/3 - w +1
    dw = (v +a -b*w)/tau
    return [dv,dw]

a = .51
b=.8
tau=2
I=1
init=[0,1]
const=(a,b,tau,I)
const2=(0.5561093 , 0.8056585 , 1.9398959 , 0.81629777 )
t  = np.linspace(0,300,3000)
X = odeint(fitz_nagumo,init,t,const)
X2=odeint(fitz_nagumo,init,t,const2)

fig,ax = plt.subplots(nrows=2,sharex=True,sharey=True)
ax[0].plot(X)
ax[1].plot(X2)
