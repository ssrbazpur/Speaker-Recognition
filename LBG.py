# -*- coding: utf-8 -*-
"""
Created By:

@author: SIMRANJEET ,KANISHA,BALJINDER AND KULSOOM
"""

#speaker Vector Quantization codebook using LBG algorithm
from __future__ import division
import numpy as np


#LBG Algorithm Implementation
def lbg(features, M):
    distortion = 1
    Centroid_num = 1 
    fraction = 0.01
    codebook = np.mean(features, 1)
    
    while Centroid_num < M:
        
        #split the size of codebook
        new_codebook = np.empty((len(codebook), Centroid_num*2))
        if Centroid_num == 1:
            new_codebook[:,0] = codebook*(1+fraction)
            new_codebook[:,1] = codebook*(1-fraction)
        else:    
            for i in range(Centroid_num):
                new_codebook[:,2*i] = codebook[:,i] * (1+fraction)
                new_codebook[:,2*i+1] = codebook[:,i] * (1-fraction)
        
        codebook = new_codebook
        Centroid_num = np.shape(codebook)[1]
        D = EUDistance(features, codebook)
        
        
        while np.abs(distortion) > fraction:
       	    
            prev_distance = np.mean(D)
            #nearest neighbour search
            nearest_codebook = np.argmin(D,axis = 1)
        
            #cluster vectors and find new centroid
            for i in range(Centroid_num):
                codebook[:,i] = np.mean(features[:,np.where(nearest_codebook == i)], 2).T 
            codebook = np.nan_to_num(codebook)    
            
            D = EUDistance(features, codebook)
            distortion = (prev_distance - np.mean(D))/prev_distance
            
    return codebook


#calculate Euclidean distance
def EUDistance(d,c):
    p = np.shape(c)[1]
    n = np.shape(d)[1]
    
    distance = np.empty((n,p))
    
    if n<p:
        for i in range(n):
            copies = np.transpose(np.tile(d[:,i], (p,1)))
            distance[i,:] = np.sum((copies - c)**2,0)
    else:
        for i in range(p):
            copies = np.transpose(np.tile(c[:,i],(n,1)))
            distance[:,i] = np.transpose(np.sum((d - copies)**2,0))
            
    distance = np.sqrt(distance)
    return distance
        
"""
REFERENCE:
https://mkonrad.net/projects/gen_lloyd.html
"""            
