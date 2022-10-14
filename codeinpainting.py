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
import pdb

#%% SECTION 2 : génération de Ω et δΩ

im = skio.imread('lena.tif')
newim = im
height, width = im.shape

# Génération de Ω

def getOmega(startheight, endheight, endwidth):
    omega = np.zeros((height,width), dtype=int)
    omega[startheight:endheight, startheight:endwidth] = 1
    return omega

# Génération de δΩ

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

# On vérifie si Ω est vide 

def isOmegaEmpty(omega):
    return (omega.sum() == 0)

#%% SECTION 3 : Variables globales, initialisation

omega0 = getOmega(20, 50, 50)
currentOmega = getOmega(20, 50, 50)
currentOmegaBarre = oppositeMask(currentOmega)
currentDeltaOmega = getDeltaOmega(currentOmega)


# CM est la matrice des confidence

CM = oppositeMask(omega0).astype(float)

# PM est la matrice des priorités 

PM = np.zeros((height, width), dtype = float)

# Taille du patch

# A MODIFIER
size = 7


#%% SECTION 4 : Fonctions utiles pour l'algorithme

# Fonctions de test d'appartenance

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

# On construit une image privée de la partie Ω

def imSansOmega(im, currentOmega):
    return np.multiply(im, currentOmega)

# On calcule le patch associé à une position p = [i,j]

def patch(position, im):
    im2 = imSansOmega(im, currentOmega)
    P = np.zeros((size,size), dtype = float)
    
    #P=im2[(position[0]-halfpatchsize):(position[0]+halfpatchsize),(position[1]-halfpatchsize):(position[1]+halfpatchsize)]
    
    for i in range (size):
        for j in range (size):
            P[i][j]= im2[position[0]-int(size/2)+i][position[1]-int(size/2)+j]
                        
    return (P,position)

# On calcule le gradient

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

# On définit ici un masque associé à un patch de position p = [i,j]

def maskFromPatch(p):
    mask = np.zeros((size, size), dtype = int)
    
    for i in range(size):
        for j in range(size):
            iabs = p[0]-int(size/2)+i ; jabs = p[1]-int(size/2)+j
            pabs = [iabs, jabs]
            if (isInCurrentOmega(pabs)== False):
                mask[i][j]=1
                
    return mask

# On calcule la distance entre deux patch définis par leurs position p et q 

def distance(p, q):
    
    maskP = maskFromPatch(p)
    
    patchP = patch(p, im) ; patchQ = patch(q, im)
    
    d = np.multiply(np.multiply(patchP[0]-patchQ[0],patchP[0]-patchQ[0]),maskP)
    s = d.sum()
    
    return s


#%% SECTION 5 : Algorithme global

def calculConfidence(p):
    
    print("p: ",p)
    assert isInCurrentOmega(p)
    
    ip = p[0] ; jp = p[1]
    for i in range (size):
        for j in range (size) :
            q = [p[0]-int(size/2)+i, p[1]-int(size/2)+j]
            if (isInCurrentOmegaBarre(q)):
                CM[ip][jp] += CM[q[0]][q[1]]
    CM[ip][jp] = CM[ip][jp]/(float(size*size))
    
    return CM[ip][jp]
    
def dataTerm(p):
    return 1

def priority(p):
    ip = p[0] ; jp = p[1]
    CM[ip][jp] = calculConfidence(p)
    PM[ip][jp] = CM[ip][jp]*dataTerm(p)
    return PM[ip][jp]

def inpainting(im, omega):
    
    # On définit δΩ à partir du Ω initial
    
    deltaOmega = getDeltaOmega(omega0) 
    plt.imshow(deltaOmega),plt.show()
    
    # Tant que Ω ≠ ∅, on poursuit l'algorithme :
    
    while (not isOmegaEmpty(currentOmega)):
        
        # On vérifie si initialement on a bien δΩ ≠ ∅
        
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
        
        # On vérifie que patch de q n'a aucun pixel dans Ω
        
        dMin = 100000000
        qExamplar = [-1,-1]
        
        for i in range(int(size/2), height - int(size/2)):
            for j in range(int(size/2), width - int(size/2)):
                q = [i,j] ; boo = True
                for k in range(size):
                    for l in range(size):
                        x = q[0]-int(size/2)+k ; y = q[1]-int(size/2)+l
                        if (isInOmega([x,y])):
                            boo = False
         
        # Si patchQ n'a aucun pixel dans Ω, on calcule sa distance et 
        # on la compare à dMin                 
         
                if (boo):
                    d = distance(pMax, q)
                    if d < dMin:
                        dMin = d
                        qExamplar = q
                        
        # On copie dans les pixels qui sont dans Ω et patchP
        # la valeur des pixels associés dans patchQ 
        
        for i in range(size):
            for j in range(size):
                x = pMax[0]-int(size/2)+i ; y = pMax[1]-int(size/2)+j
                #print("p0: ",pMax[0])
                #print("p1 : ",pMax[1])
                #print("x : ",x)
                #print("y : ",y)
                if (isInCurrentOmega([x,y])):
                    newim[x][y]=patch(qExamplar,im)[0][i][j]
        
        # On met à jour la confidence
        
        for i in range(size):
            for j in range(size):
                x = pMax[0]-int(size/2)+i ; y = pMax[1]-int(size/2)+j
                if (isInCurrentOmega([x,y])):
                    print("x : ",x)
                    print("y : ",y)
                    calculConfidence([x,y])
                    
        # On met à jour currentOmega
        
        for i in range(size):
            for j in range(size):
                x = pMax[0]-int(size/2)+i ; y = pMax[1]-int(size/2)+j
                currentOmega[x][y] = 0 ;
                
        # On met à jour δΩ à la fin de la boucle
        
        deltaOmega = getDeltaOmega(currentOmega)
        print("Iter")
    
    return 0

inpainting(im, omega0)

imgplot = plt.imshow(newim)
plt.show()

