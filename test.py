# -*- coding: utf-8 -*-
"""
Created By:

@author: SIMRANJEET ,KANISHA,BALJINDER AND KULSOOM
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from MFCC_algorithm import mfcc
from train import training
import os

#total number of speakers and filters required 
totalspeakers = 4
nfilters = 16
#assigning the location of testing data
directory = os.getcwd() + '/test';
fname = str()
codebooks = training(nfilters)

#counter to count the number of speakers correctly identified
nCorrect_MFCC = 0


#calculating the minimum distance between neighbours
def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0]) 
        if dist < distmin:            
            distmin = dist
            speaker = k
            
    return speaker
    

#performing testing
for i in range(totalspeakers):
    fname = '/s' + str(i+1) + '.wav'
    print('Now speaker ', str(i+1), 'features are being tested')
    (fs,s) = read(directory + fname)
    mel_coefs = mfcc(s,fs,nfilters) 
    sp_mfcc = minDistance(mel_coefs, codebooks)    
    print('Speaker ', (i+1), ' in test matches with speaker ', (sp_mfcc+1), ' in train for training with MFCC')   
    if i == sp_mfcc:
        nCorrect_MFCC += 1

#calculating percentage accuracy
percentageCorrect_MFCC = (nCorrect_MFCC/totalspeakers)*100
print('Accuracy of result with MFCC is ', percentageCorrect_MFCC, '%')



    
