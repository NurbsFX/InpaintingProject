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
from extended_int import int_inf
import pdb
from skimage import color


#%% SECTION 2 : génération de Ω et δΩ

im = skio.imread('lena128.tif')
imoriginale = im
#resized_image = im.resize((64,64))
height, width = im.shape[0],im.shape[1]

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

omega0 = getOmega(40, 60, 60)
#currentOmega = getOmega(20, 50, 50)
#currentOmegaBarre = oppositeMask(currentOmega)
#currentDeltaOmega = getDeltaOmega(currentOmega)

alpha = 255

# CM est la matrice des confidence

CM = oppositeMask(omega0).astype(float)

# PM est la matrice des priorités 

PM = np.zeros((height, width), dtype = float)

# Taille du patch

# A MODIFIER
patchSize = 5
halfPatchSize = int(patchSize/2)


#%% SECTION 4 : Fonctions utiles pour l'algorithme

# Fonctions de test d'appartenance

def isInOmega(p, omega):
    ip = p[0] ; jp = p[1]
    return (omega[ip][jp]==1)

def isInCurrentOmega(p, currentOmega):
    ip = p[0] ; jp = p[1]
    return (currentOmega[ip][jp]==1)

def isInCurrentOmegaBarre(p, currentOmegaBarre):
    ip = p[0] ; jp = p[1]
    return (currentOmegaBarre[ip][jp]==1)

def isInCurrentDeltaOmega(p, currentDeltaOmega):
    ip = p[0] ; jp = p[1]
    return (currentDeltaOmega[ip][jp]==1)

# On construit une image privée de la partie Ω

def imSansOmega(im, currentOmega):
    return np.multiply(im, oppositeMask(currentOmega))

# On calcule le patch associé à une position p = [i,j]

def patch(position, im, currentOmega):
    #im2 = imSansOmega(im, currentOmega)
    #Pboucle = np.zeros((patchSize,patchSize), dtype = float)
    
    Pmatrice=im[(position[0]-halfPatchSize):(position[0]+halfPatchSize+1),(position[1]-halfPatchSize):(position[1]+halfPatchSize+1)]
    Pmatrice = Pmatrice.astype('float64')
    
    # for i in range (patchSize):
    #     for j in range (patchSize):
    #         Pboucle[i][j]= im2[position[0]-halfPatchSize+i][position[1]-halfPatchSize+j]
            
    #print("Matrice P obtenue avec boucle :",Pboucle)
    #print("Matrice P obtenue avec matrices :",Pmatrice)
              
    return (Pmatrice,position)

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

def maskFromPatch(p, omega, im):
    if (len(im.shape)==2):
        mask = np.zeros((patchSize, patchSize), dtype = float)
        
        #mask=im[(p[0]-halfPatchSize):(p[0]+halfPatchSize+1),(p[1]-halfPatchSize):(p[1]+halfPatchSize+1)]
        #oppositeMaskResized = oppositeMask(omega)[(p[0]-halfPatchSize):(p[0]+halfPatchSize+1),(p[1]-halfPatchSize):(p[1]+halfPatchSize+1)]
        
        for i in range(patchSize):
            for j in range(patchSize):
                iabs = p[0]-halfPatchSize+i ; jabs = p[1]-halfPatchSize+j
                pabs = [iabs, jabs]
                if (isInCurrentOmega(pabs, omega)== False):
                    mask[i][j]=1
    else:
        mask = np.zeros((patchSize, patchSize, im.shape[2]), dtype = float)
        
        #mask=im[(p[0]-halfPatchSize):(p[0]+halfPatchSize+1),(p[1]-halfPatchSize):(p[1]+halfPatchSize+1)]
        #oppositeMaskResized = oppositeMask(omega)[(p[0]-halfPatchSize):(p[0]+halfPatchSize+1),(p[1]-halfPatchSize):(p[1]+halfPatchSize+1)]
        
        for i in range(patchSize):
            for j in range(patchSize):
                iabs = p[0]-halfPatchSize+i ; jabs = p[1]-halfPatchSize+j
                pabs = [iabs, jabs]
                if (isInCurrentOmega(pabs, omega)== False):
                    mask[i][j]=1
    
    #maskf= np.multiply(mask, oppositeMaskResized)
                
    return mask

# On calcule la distance entre deux patch définis par leurs position p et q 

def distance(p, q, omega, im):
    
    maskP = maskFromPatch(p, omega, im)
    
    patchP = patch(p, im, omega) ; patchQ = patch(q, im, omega)
    #print("patchP[0] :", patchP[0])
    #print("patchQ[0] :", patchQ[0])
    
    d = np.multiply(np.multiply(patchP[0]-patchQ[0],patchP[0]-patchQ[0]),maskP)
    s = d.sum()
    
    return s

def visuPatch(im,p,q):
    im1=im.copy
    im1


#%% SECTION 5 : Algorithme global

def calculConfidence(p, omega):
    
    #print("p: ",p)
    assert isInCurrentOmega(p, omega)
    omegaBarre = oppositeMask(omega)
    
    ip = p[0] ; jp = p[1]
    for i in range (patchSize):
        for j in range (patchSize) :
            q = [p[0]-halfPatchSize+i, p[1]-halfPatchSize+j]
            if (isInCurrentOmegaBarre(q, omegaBarre)):
                CM[ip][jp] += CM[q[0]][q[1]]
    CM[ip][jp] = CM[ip][jp]/(float(patchSize*patchSize))
    
    return CM[ip][jp]
    
def dataTerm(imgrad,p):
    
    
    grad = [] ; isophote = [] ; normalp = []
    
    imgradx = imgrad[0]
    imgrady = imgrad[1]

    pgradx = imgradx[(p[0]-halfPatchSize):(p[0]+halfPatchSize),(p[1]-halfPatchSize):(p[1]+halfPatchSize)]
    pgrady = imgrady[(p[0]-halfPatchSize):(p[0]+halfPatchSize),(p[1]-halfPatchSize):(p[1]+halfPatchSize)]
    
    #print(pgradx)
    #print(np.max(pgradx))

    grad.append(np.max(pgradx))
    grad.append(np.max(pgrady))
    
    isophote.append(-grad[1])
    isophote.append(grad[0]) 
    
    normalp.append(imgradx[p[0],p[1]])
    normalp.append(imgrady[p[0],p[1]])
    
    d = np.dot(isophote,normalp)/alpha
    
    return d

def priority(im, p, omega):
    ip = p[0] ; jp = p[1]
    CM[ip][jp] = calculConfidence(p, omega)
    PM[ip][jp] = CM[ip][jp]*dataTerm(im, p)
    return PM[ip][jp]

def inpainting(im, omega):
    
    newim = im.copy()
    compteur = 1
    
    
    # On définit δΩ à partir du Ω initial
    
    currentOmega = omega
    deltaOmega = getDeltaOmega(omega)
    
    #plt.imshow(deltaOmega),plt.show()
    
    # On colorie les pixels représentant Ω
    
    for i in range(height):
        for j in range(width):
            if (isInOmega([i,j], omega)):
                newim[i,j]=0
    
    plt.imshow(newim), plt.title("Image originale avec partie manquante "), plt.show()
    
    # Tant que Ω ≠ ∅, on poursuit l'algorithme :
    
    while (not isOmegaEmpty(currentOmega)):
        
        
        # On vérifie si initialement on a bien δΩ ≠ ∅
        
        nullMatrix = np.zeros((height,width), dtype =int)
        if ((deltaOmega == nullMatrix).all()):
            print("DeltaOmega est vide")
            return 0
        
        
        im2=im.copy()
        if(len(im2.shape) == 3):
            im2 = color.rgb2gray(im)
        im2[currentOmega>0]=-10000000
        
        im_grad = np.gradient(im2)
        
        
        # On calcule le patch de priorité Max
        
        priorityMax = 0
        pMax = [-1,-1]
        
        for i in range(height):
            for j in range(width):
                p = [i,j]
                if (isInCurrentDeltaOmega(p, deltaOmega)):
                    pValue = priority(im_grad, p, currentOmega)
                    if (pValue > priorityMax):
                        priorityMax = pValue
                        pMax = p
        
        print("pMax :", pMax)
        
        # On vérifie que patch de q n'a aucun pixel dans Ω
        
        dMin = 100000000
        qExamplar = [-1,-1]
        
        for i in range(halfPatchSize, height - halfPatchSize):
            for j in range(halfPatchSize, width - halfPatchSize):
                q = [i,j] ; boo = True
                for k in range(patchSize):
                    for l in range(patchSize):
                        x = q[0]-halfPatchSize+k ; y = q[1]-halfPatchSize+l
                        if (isInOmega([x,y],omega)):
                            boo = False
         
        # Si patchQ n'a aucun pixel dans Ω, on calcule sa distance et 
        # on la compare à dMin                 
         
                if (boo):
                    d = distance(pMax, q, currentOmega, newim)
                    #print("d(p,q) :", d)
                    if d < dMin:
                        dMin = d
                        #print("dMin :", dMin)
                        qExamplar = q
                        #print("qExamplar :", qExamplar)
                        
        QExamplar = np.zeros((height, width), dtype = float)
        for i in range (patchSize):
            for j in range (patchSize):
                QExamplar[qExamplar[0]-halfPatchSize+i][qExamplar[1]-halfPatchSize+j]= 1
                
        #plt.imshow(QExamplar), plt.title("Emplacement du patch q à la boucle {}".format(compteur)), plt.show()
        print("dMin : ", dMin)
        print("qExamplar : ", qExamplar)
        print("Value :", newim[qExamplar[0],qExamplar[1]])
                   
        # On copie dans les pixels qui sont dans Ω et patchP
        # la valeur des pixels associés dans patchQ 
        
        
        
        for i in range(patchSize):
            for j in range(patchSize):
                x = pMax[0]-halfPatchSize+i ; y = pMax[1]-halfPatchSize+j
                #print("p0: ",pMax[0])
                #print("p1 : ",pMax[1])
                #print("x : ",x)
                #print("y : ",y)
                if (isInCurrentOmega([x,y], currentOmega)):
                    #print(patch(qExamplar,im, currentOmega)[0].shape)
                    #print("Ancienne valeur de newim[x][y] :", newim[x][y])
                    newim[x][y]=patch(qExamplar,im, currentOmega)[0][i][j]
                    #print("Nouvelle valeur de newim[x][y] :", newim[x][y])
                    #plt.imshow(newim), plt.title("Lena MAJ"), plt.show()
        
        # On met à jour la confidence
        
        for i in range(patchSize):
            for j in range(patchSize):
                x = pMax[0]-halfPatchSize+i ; y = pMax[1]-halfPatchSize+j
                if (isInCurrentOmega([x,y], currentOmega)):
                    #print("x : ",x)
                    #print("y : ",y)
                    calculConfidence([x,y], currentOmega)
                    
        # On met à jour currentOmega
        
        for i in range(patchSize):
            for j in range(patchSize):
                x = pMax[0]-halfPatchSize+i ; y = pMax[1]-halfPatchSize+j
                currentOmega[x][y] = 0 ;
                
        # On met à jour δΩ à la fin de la boucle
        
        deltaOmega = getDeltaOmega(currentOmega)
        plt.imshow(newim), plt.title("Image après la boucle {}".format(compteur)), plt.show()
        #plt.imshow(currentOmega), plt.title("Ω après la boucle {}".format(compteur)), plt.show()

        print(" ")
        print("FIN DE LA BOUCLE {}".format(compteur))
        print(" ")
        compteur +=1
    
    imgplot = plt.imshow(newim), plt.title('Image modifiée avec un pacth de taille {}'.format(patchSize))
    plt.show()
    imgplot = plt.imshow(im), plt.title('Image originale')
    plt.show()
    return 0

inpainting(im, omega0)

#imgplot = plt.imshow(newim)
#plt.show()

#plt.imshow(grady(im)), plt.show()

