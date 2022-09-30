#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:04:32 2022

@author: brunokalfa
"""

#%% SECTION 1 inclusion de packages externes 

import numpy as np
import matplotlib.pyplot as plt
from skimage import io as io


#%% SECTION 2 Fonctions utiles

def patch(size, position):
    l = []
    l.append(size, position)
    return l

def gradx(im):
    "renvoie le gradient dans la direction x"
    imt=np.float32(im)
    gx=0*imt
    gx[:,:-1]=imt[:,1:]-imt[:,:-1]
    return gx

def grady(im):
    "renvoie le gradient dans la direction y"
    imt=np.float32(im)
    gy=0*imt
    gy[:-1,:]=imt[1:,:]-imt[:-1,:]
    return gy

def confidence(p):
    return 0
    
def data(p):
    return 0

def priority(p):
    return confidence(p)*data(p)