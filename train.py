# -*- coding: utf-8 -*-
"""
Created By:

@author: SIMRANJEET ,KANISHA,BALJINDER AND KULSOOM
"""

from __future__ import division
from LBG import lbg
import numpy as np
from scipy.io.wavfile import read
from MFCC_algorithm import mfcc
import matplotlib.pyplot as plt

import os


#training function to train the speaker in order to carry out speaker recognition
def training(nfiltbank):
    nSpeaker = 4
    nCentroid = 4
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'
        print('Now speaker ', str(i+1), 'features are being trained' )
        (fs,s) = read(directory + fname)
        #REMOVE COMMENTS TO PLOT THE INPUT SPEAKER WAVE
        #audio = (fs,s)[1]
        # plot the first 1024 samples
        #plt.plot(audio[0:1024])
        # label the axes
        # plt.ylabel("Amplitude")
        # plt.xlabel("Time")
        # set the title  
        
        #plt.title("Sample Wav")
        # display the plot
        #plt.show()
        mel_coeff = mfcc(s, fs, nfiltbank)
        codebooks_mfcc[i,:,:] = lbg(mel_coeff, nCentroid)
        
        plt.figure(i)
        for j in range(nCentroid):
            plt.subplot(211)
            plt.stem(codebooks_mfcc[i,:,j])
            plt.xlabel('Number of features')
            plt.ylabel('MFCC')
            plt.title('Codebook for speaker ' + str(i+1))
    
    plt.show()
    print('Training completed')
    
    return (codebooks_mfcc)
    
    
