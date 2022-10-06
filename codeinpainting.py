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
newim = im
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

def isOmegaEmpty(omega):
    return (omega.sum() == 0)

#%% SECTION 3 : Variables globales, initialisation

omega0 = getOmega(100, 150, 150)
currentOmega = getOmega(100, 150, 150)
currentOmega
currentOmegaBarre = oppositeMask(currentOmega)
currentDeltaOmega = getDeltaOmega(currentOmega)

CM = oppositeMask(omega0)
PM = np.zeros((height, width), dtype = int)

# Taille du patch
size = 7

def isInOmega(p):
    ip = p[0] ; jp = p[1]
    return (omega0[ip][jp]==1)

def isInCurrentOmega(p):
    ip = p[0] ; jp = p[1]
    return (currentOmega[ip][jp]==1)

def isInCurrentOmegaBarre(p):
    ip = p[0] ; jp = p[1]
    return (currentOmegaBarre[ip][jp]==1)

def isInCurrentDeltaOmega(p):
    ip = p[0] ; jp = p[1]
    return (currentDeltaOmega[ip][jp]==1)


#%% SECTION 4 : Fonctions utiles pour l'algorithme

def imSansOmega(im, currentOmega):
    return np.multiply(im, currentOmega)

def patch(position, im):
    im2 = imSansOmega(im, currentOmega)
    P = np.zeros((size,size), dtype = int)
    
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

# Métrique de patch

def maskFromPatch(p):
    mask = np.zeros((size, size), dtype = int)
    
    for i in range(size):
        for j in range(size):
            iabs = p[0]-int(size/2)+i ; jabs = p[1]-int(size/2)+j
            pabs = [iabs, jabs]
            if (isInCurrentOmega(pabs)== False):
                mask[i][j]=1
                
    return mask


def distance(p, q):
    
    maskP = maskFromPatch(p)
    
    patchP = patch(p, im) ; patchQ = patch(q, im)
    
    d = np.multiply(np.multiply(patchP[0]-patchQ[0],patchP[0]-patchQ[0]),maskP)
    s = d.sum()
    
    return s


#%% SECTION 5 : Algorithme global

def calculConfidence(p):
    
    assert isInCurrentDeltaOmega(p)
    
    ip = p[0] ; jp = p[1]
    for i in range (size):
        for j in range (size) :
            q = [p[0]-int(size/2)+i, p[1]-int(size/2)+j]
            if (isInCurrentOmegaBarre(p)):
                CM[ip][jp] += CM[q[0]][q[1]]
    CM[ip][jp] = CM[ip][jp]/(size*size)
    
    return CM[ip][jp]
    
def dataTerm(p):
    return 1

def priority(p):
    ip = p[0] ; jp = p[1]
    CM[ip][jp] = calculConfidence(p)
    PM[ip][jp] = CM[ip][jp]*dataTerm(p)
    return PM[ip][jp]

def inpainting(im, omega):
    
    deltaOmega = getDeltaOmega(omega0) 
    
    while (not isOmegaEmpty(currentOmega)):
        
        nullMatrix = np.zeros((height,width), dtype =int)
        if ((deltaOmega == nullMatrix).all()):
            print("DeltaOmega est vide")
            return 0
        
        # On calcule le patch de priorité Max
        
        priorityMax = 0
        pMax = [-1,-1]
        
        for i in range(height):
            for j in range(width):
                p = [i,j]
                if (isInCurrentDeltaOmega(p)):
                    pValue = priority(p)
                    if (pValue > priorityMax):
                        priorityMax = pValue
                        pMax = p
        
        # On vérifie que patchQ n'a aucun pixel dans Omega
        
        dMin = 100000000
        qExamplar = [-1,-1]
        
        for i in range(height):
            for j in range(width):
                q = [i,j] ; boo = True
                for k in range(size):
                    for l in range(size):
                        x = q[0]-int(size/2)+i ; y = q[1]-int(size/2)+j
                        if (isInOmega([x,y])):
                            boo = False
                if (boo):
                    d = distance(pMax, q)
                    if d < dMin:
                        dMin = d
                        qExamplar = q
                        
        # On copie dans patchP les éléments de patchQ 
        
        for i in range(size):
            for j in range(size):
                x = p[0]-int(size/2)+i ; y = p[1]-int(size/2)+j
                if (isInCurrentOmega([x,y])):
                    newim[x][y]=patch(qExamplar)[0][i][j]
        
        # On met à jour la confidence
        
        for i in range(size):
            for j in range(size):
                x = p[0]-int(size/2)+i ; y = p[1]-int(size/2)+j
                if (isInCurrentOmega([x,y])):
                    calculConfidence([x,y])
                    
        # On met à jour currentOmega
        
        for i in range(size):
            for j in range(size):
                x = p[0]-int(size/2)+i ; y = p[1]-int(size/2)+j
                currentOmega[x][y] = 0 ;
                
        # On met à jour deltaOmega 
        
        deltaOmega = getDeltaOmega(currentOmega)
    
    return 0

inpainting(im, omega0)

imgplot = plt.imshow(newim)
plt.show()

