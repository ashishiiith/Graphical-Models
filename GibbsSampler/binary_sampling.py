#Author: Ashish Jain
#CMPSCI - 688 PGM, Gibbs Sampling for binary Images
#How to Run code? python binary_sampling.py <arg1> <arg2>
#arg1 = WP parameter value
#arg2 = WL parameter value 
#Code for question 1

import sys, os, re
import math
from numpy import *
from scipy import *
from PIL import Image
import pylab as plt
WL = None
WP = None
T  = 100 #Number of Iterations to run
H  = 50 #Height of Image
W  = 50 #Width of Image

xij = zeros([50, 50])
yij = zeros([50, 50])
I   = zeros([50, 50])
MAE=[]
def draw_image(matrix, fname):
    ''' Using PIL library to draw images
    '''
    image = Image.fromarray((matrix*255).astype(uint8))
    image.show()
    name = str(fname)
    image.save(name)

def plot(y):
    ''' Using Pylab to plot iteration vs MAE curve
    '''
    x = range(len(y))
    plt.plot(x, y)
    plt.axis([0, len(y) + 10, 0.001, 0.005])
    plt.xlabel('Iterations')
    plt.ylabel('MAE')
    plt.title('MAE vs Iterations')
    wl = sys.argv[2]
    wp = sys.argv[1]
    f = str(T) + "_" + str(wp) + "_" + str(wl) + "_stripes_graph.png"
    print "Saving figure.."
    plt.savefig(f)
    plt.ion()
    plt.show()

def load_pixels(file_name):
    
    index = 0
    tmp = zeros([50 ,50])
    for line in open(file_name, "r"):
        tmp[index] = map(lambda x : int(float(x)), line.strip().split()) 
        index+=1
    return tmp

def neighbors(x, y, val):
    '''Computing neigbors of a given (x, y) position
    ''' 
    s = 0
    if x-1 >= 0:
        s+=(val == yij[x-1][y])
    if y-1 >= 0:
        s+=(val == yij[x][y-1])
    if x+1 <=49:
        s+=(val == yij[x+1][y])
    if y+1 <=49:
        s+=(val == yij[x][y+1])
    return s

def conditional_distribution(x, y):
        
    s0 = WP*neighbors(x, y, 0) + WL*(xij[x][y] == 0)
    s1 = WP*neighbors(x, y, 1) + WL*(xij[x][y] == 1)
    prob = float(exp(s1))/float(exp(s1) + exp(s0))
    return prob 

def compute_posterior_meanImage():

    global WL
    global WP
    global yij
    global xij
    global H
    global W
    global MAE
    yijt = zeros([50, 50])
    for iteration in xrange(0, T):
       error = 0.0
       for x in xrange(0, 50):
           for y in xrange(0, 50):  
                prob  = conditional_distribution(x, y)
                alpha =  random.uniform() #generation random alpha
                if prob > alpha: 
                    yij[x][y] = 1 
                else: 
                    yij[x][y] = 0
                yijt[x][y] += yij[x][y] #Saving sampled pixel
                error += absolute(yijt[x][y]/float(iteration+1) - I[x][y])
       MAE.append(0)
       MAE[iteration] = error/float(H*W) #MAE computation
       print "Iteration complete : " +  str(iteration) + " " +  str(MAE[iteration])
    
def main():
 
    global xij
    global yij
    global I
    global WL
    global WP
    global MAE
    WP = int(sys.argv[1]) #favors smoothness
    WL = int(sys.argv[2]) #favors being close to noisy image
    xij = load_pixels("Data/stripes-noise.txt")
    yij = copy(xij) #initialize yij = xij
    I   = load_pixels("Data/stripes.txt")    
    compute_posterior_meanImage()
    draw_image(yij, "stripes-noisyy.png")

if __name__ == "__main__":
    main()

