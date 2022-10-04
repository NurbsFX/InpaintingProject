#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 15:04:32 2022

@author: brunokalfa
"""

#%% SECTION 1 : Inclusion de packages externes 

import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from scipy import ndimage

#%% SECTION 2 : génération de δΩ

im = skio.imread('lena.tif')
height, width = im.shape

def getOmega(startheight, endheight, endwidth):
    omega = np.zeros((height,width), dtype=int)
    omega[startheight:endheight, startheight:endwidth] = 1
    return omega

def getDeltaOmega(omega):
    deltaomega = omega - ndimage.binary_erosion(omega).astype(omega.dtype)
    return deltaomega 

# Renvoie le mask complémentaire du mask entré en argument
def oppositeMask(mask):
    nblignes, nbcolonnes = mask.shape
    newmask = np.zeros((nblignes,nbcolonnes), dtype =int)
    
    for i in range(nblignes):
        for j in range(nbcolonnes):
            newmask[i][j] = 1 - mask[i][j]
    
    return newmask


#%% SECTION 3 : Variables globales, initialisation

omega0 = getOmega(100, 150, 150)
currentOmega = getOmega(100, 150, 150)
currentOmega
currentOmegaBarre = oppositeMask(currentOmega)
currentDeltaOmega = getDeltaOmega(currentOmega)

CM =oppositeMask(omega0)

# Taille du patch
size = 7

def isInCurrentOmega(p):
    ip = p[0], jp = p[1]
    return (currentOmega[ip][jp]==1)

def isInCurrentOmegaBarre(p):
    ip = p[0], jp = p[1]
    return (currentOmegaBarre[ip][jp]==1)

def isInCurrentDeltaOmega(p):
    ip = p[0], jp = p[1]
    return (currentDeltaOmega[ip][jp]==1)


#%% SECTION 4 : Fonctions utiles pour l'algorithme

def imSansOmega(im, currentOmega):
    return np.multiply(im, currentOmega)

def patch(position, im):
    im2 = imSansOmega(im, currentOmega)
    P = np.zero(size,size)
    
    for i in range (size) :
        for j in range (size) :
            P[i][j]= im2[position[0]-int(size/2)+i][position[1]-int(size/2)+j]
                        
    return (P,position)

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


#%% SECTION 5 : Algorithme global

def calculConfidence(p):
    assert isInCurrentDeltaOmega(p)
    ip = p[0], jp = p[1]
    for i in range (size):
        for j in range (size) :
            q = [p[0]-int(size/2)+i, p[1]-int(size/2)+j]
            if (isInCurrentOmegaBarre(p)):
                CM[ip][jp] += CM[q[0]][q[1]]
    return CM[ip][jp]/(size*size)
    
def data(p):
    return 1

def priority(p):
    ip = p[0], jp = p[1]
    return CM[ip][jp]*data(p)

def inpainting(im, omega):
    return 0

