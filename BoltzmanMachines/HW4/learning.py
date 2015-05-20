#Author - Ashish Jain, PGM CMPSCI 688
#How to Run code? python learning.py

''' This code learns RBM for given raw pixels values of image. This is code for Question 3.
'''
import os, sys
import pandas as pd
import math
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import matplotlib.cm as cm

K = 400 # no. of hidden units H
T = 50 # no. of iterations for sampling
D = 784 # no. of observed units X
C = 100 #no. of Gibbs Chain
B = 100 #no. of batches
WP = None
WB = None
WC = None
h = zeros([C, K])

def load_data():

    print "Loading data.."
    data = pd.read_csv('Data/MNISTXtrain.txt',sep=' ', dtype=int)
    #print data.values
    return data.values

def probability(x):

    e = exp(x)
    return e/(1+e)

def RBMLearn(NB, Lambda, alpha):
    global WP, WB, WC, h    
    data = load_data()
    print "Learning Started.." 
 
    #initialize the Gibbs Chain
    h = random.randint(0, 2, (C,K))
    
    #initialize the parameters 
    WB = random.normal(0, 0.1, K)
    WC = random.normal(0, 0.1, D)
    WP = random.normal(0, 0.1, (D,K))
    
    for t in xrange(0, T):
        print "t: " + str(t)  
        for b in xrange(0, B):

            #Compute positive gradient contribution from each data case in batch b
            xnd = data[b*NB:(b+1)*NB, :] 
            pk = probability(dot(xnd, WP) + WB) 
            gWCp = sum(xnd, axis=0)
            gWBp = sum(pk, axis=0)
            gWPp = dot(xnd.T, pk)
        
            #Compute negative gradient contribution from each chain and sample states
            u_matrix =  random.random((C, D))
            Pxcd =  probability(dot(h, WP.T) + WC)
            xcd = (Pxcd > u_matrix).astype(int)
            
            u_matrix =  random.random((C, K))
            Phck = probability(dot(xcd, WP) + WB)
            h =  (Phck > u_matrix).astype(int)
            gWCn = sum(xcd, axis=0)
            gWBn = sum(Phck, axis=0)
            gWPn = dot(xcd.T, Phck)
 
            #Take a gradient step for each parameter in the model 
            
            WC = WC + alpha*(gWCp/float(NB) - gWCn/float(C) - Lambda*WC)
            WB = WB + alpha*(gWBp/float(NB) - gWBn/float(C) - Lambda*WB)
            WP = WP + alpha*(gWPp/float(NB) - gWPn/float(C) - Lambda*WP)        
    savetxt('WP.txt', WP)
    savetxt('WB.txt', WB)
    savetxt('WC.txt', WC)
    savetxt('h.txt', h)
    print "Model Learned.."

def displayImage():

    #display last 100 samples of 100 Chains 
    global WP
    global WC
    global h
    u = random.random((C, D)) 
    prob =  probability(dot(h, WP.T) + WC)
    xnd = (prob > u).astype(int)
    fig = plt.figure()
    gs = gridspec.GridSpec(10,10)
    img = [plt.subplot(gs[i]) for i in range(C)]
    plt.setp([a.get_xticklabels() for a in img], visible=False)
    plt.setp([a.get_yticklabels() for a in img], visible=False)
    for i in range(C):
        img[i].imshow(xnd[i].reshape(28, 28), cmap = cm.Greys_r)
    plt.show()    

def ShowImage():

    WP_L = loadtxt("WP.txt")
    print WP_L.shape
    figure = plt.figure()
    gs = gridspec.GridSpec(20,20)
    gs.update(wspace=0.0, hspace=0.0)
    img = [plt.subplot(gs[i]) for i in range(400)]

    for i in xrange(0, K):
        img[i].imshow(WP_L[:,i].reshape(28, 28), cmap = cm.Greys_r)
        img[i].axes.get_xaxis().set_visible(False)
        img[i].axes.get_yaxis().set_visible(False)
    plt.show()

def main():    

    RBMLearn(600, 0.0001, 0.1)
    displayImage() 
    ShowImage()
if __name__ == "__main__":

    main()
