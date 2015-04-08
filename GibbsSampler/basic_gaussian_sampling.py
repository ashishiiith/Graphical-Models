#Author: Ashish Jain
#CMPSCI - 688 PGM, Gibbs Sampling for gaussian distribution
#How to Run code? python basic_gaussian_sampling.py <arg1> <arg2>
#arg1 = WP parameter value
#arg2 = WL parameter value
#Code for Question 2.a 

import sys, os, re
import math
from numpy import *
from scipy import *
from PIL import Image
import matplotlib.pyplot as plt

WL = None
WP = None
T  = 100 #Number of Iterations
H  = 50 #Height of Image
W  = 50 #Width of Image

xij = zeros([50, 50]) #Noisy image
yij = zeros([50, 50]) #Sampled pixel image
I   = zeros([50, 50]) #Original image

def draw_image(matrix, fname):
    ''' Using PIL library to draw images
    '''
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

    ''' Summing up the pixel values of the neigbors
    ''' 
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

    ''' Computing variance square according to equation in assignment
    '''
    denom = WP*((x-1>=0) + (y-1>=0) + (x+1 <= 49) + (y+1<=49)) +  WL
    variance = 1.0/(2.0*float(denom))
    return float(variance)

def compute_mean(x, y):
    ''' Computing mean according to given equation in assignment
    '''
    numerator = (WL*xij[x][y]) + WP*(neighbors(x,y))
    mean = float(numerator)*float(compute_variance(x, y)*2)
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
                yij[x][y] = compute_mean(x, y) + z*sqrt(compute_variance(x, y)) #computing new pixel value
                yijt[x][y] += yij[x][y] #saving sampled pixels
                error += absolute(yijt[x][y]/float(iteration+1) - I[x][y]) #computing sampled image error
       MAE.append(0)
       MAE[iteration] = error/float(H*W)
       print "Iteration complete : " +  str(iteration) + " " +  str(MAE[iteration])

def main():
 
    global xij
    global yij
    global I
    global WL
    global WP

    WP = int(sys.argv[1]) #favors smoothness
    WL = int(sys.argv[2]) #favors noisy image
    xij = load_pixels("Data/swirl-noise.txt")
    yij = copy(xij) #initialize yij = xij
    I   = load_pixels("Data/swirl.txt") #loading origial image
    compute_posterior_meanImage()
    draw_image(yij, "denoised_swirl.png")

if __name__ == "__main__":
    main()

