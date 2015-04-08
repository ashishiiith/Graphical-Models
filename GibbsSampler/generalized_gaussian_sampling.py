#Author: Ashish Jain
#CMPSCI - 688 PGM, Gibbs Sampling for gaussian distribution
#How to Run code? python generalized_gaussian_sampling.py  <arg1> <arg2>
#arg1 = WP parameter value
#arg2 = WL parameter value 
#Code for question 2.c

import sys, os, re
import math
from numpy import *
from scipy import *
from PIL import Image

WL = None
WP = None
WPR = zeros([49, 50]) #Storing vertical WP weights between pixels
WPC = zeros([50, 49]) #Storing horizontol WP weights between pixels
T  = 100 #Number of Iterations
H  = 50  #Heights of Image
W  = 50  #Width of Image

xij = zeros([50, 50])
yij = zeros([50, 50])
I   = zeros([50, 50])


def draw_image(matrix, fname):

    image = Image.fromarray((matrix*255).astype(uint8))
    image.show()
    name = str(fname)
    image.save(name)

def load_pixels(file_name):
    
    index = 0
    tmp = zeros([50 ,50])
    for line in open(file_name, "r"):
        tmp[index] = map(lambda x : float(x), line.strip().split())
        index+=1 
    return tmp

def neighbors(x, y):
 
    s = float(0.0)
    if x-1 >= 0:
        s+= yij[x-1][y]
    if y-1 >= 0:
        s+= yij[x][y-1]
    if x+1 <=49:
        s+= yij[x+1][y]
    if y+1 <=49:
        s+= yij[x][y+1]
    return s

def compute_variance(x, y):

    W =  float(WL)
    if x-1 >= 0:
        W+=WPR[x-1][y]
    if y-1 >= 0:
        W+=WPC[x][y-1]
    if x+1<=49:
        W+=WPR[x][y]
    if y+1<=49:
        W+=WPC[x][y]
    
    variance = 1.0/float(2*W)
    return float(variance)

def compute_mean(x, y):

    Num = float(WL*xij[x][y])
    if x-1 >= 0:
        Num+=WPR[x-1][y]*yij[x-1][y]
    if y-1 >= 0:
        Num+=WPC[x][y-1]*yij[x][y-1]
    if x+1<=49:
        Num+=WPR[x][y]*yij[x+1][y]
    if y+1<=49:
        Num+=WPC[x][y]*yij[x][y+1]
    mean = float(Num)*float(compute_variance(x, y)*2)
    return mean

def compute_posterior_meanImage():

    global WL
    global WP
    global yij
    global xij
    global H
    global W
    yijt = zeros([50, 50])
    MAE = []
    for iteration in xrange(0, T):
       error = 0.0
       for x in xrange(0, 50):
           for y in xrange(0, 50):  
                z =  random.normal(0, 1)
                yij[x][y] = compute_mean(x, y) + z*sqrt(compute_variance(x, y))
                yijt[x][y] += yij[x][y]
                error += absolute(yijt[x][y]/float(iteration+1) - I[x][y])
       MAE.append(0)
       MAE[iteration] = error/float(H*W)
       print "Iteration complete : " +  str(iteration) + " " + str(MAE[iteration])

def set_general_weights():

    ''' Computing generalized weights of WP parameter
    '''
    global WPR
    global WPC

    for y in range(50):
        for x in range(49):
            WPR[x][y] = float(WP) / float(0.01 + (xij[x][y] - xij[x+1][y])**2) 
   
    for x in range(50):
        for y in range(49):
            WPC[x][y] = float(WP) / float(0.01 + (xij[x][y] - xij[x][y+1])**2)

def main():
 
    global xij
    global yij
    global I
    global WL
    global WP

    WP = float(sys.argv[1]) #favors smoothness
    WL = float(sys.argv[2]) #favors noisy image
    xij = load_pixels("Data/swirl-noise.txt")
    yij = copy(xij) #initialize yij = xij
    set_general_weights()
    I   = load_pixels("Data/swirl.txt")    
    compute_posterior_meanImage()
    draw_image(yij, "denoised_swirl_c.png")    

if __name__ == "__main__":
    main()

