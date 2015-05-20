#Author: Ashish Jain, PGM CMPSCI 688
#How to Run Code? python inference.py <number of chains>
import os, sys
import math
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import matplotlib.backends.backend_pdf
import png
import matplotlib.cm as cm

WP = zeros([784, 100])
WB = zeros([1, 100])
WC = zeros([1, 784])

flag = None
last_sample = []
K = None # no. of hidden units H
S = None # no. of iterations for sampling
D = None # no. of observed units X
E = None # energy for first five chains
e_count = 0

def load_WP(fname):

    global WP
    WP = loadtxt(fname)
 
def load_WB(filename):

    global WB
    WB = loadtxt(filename)

def load_WC(filename):
  
    global WC
    WC = loadtxt(filename)

def energy(x, h):

    first_term = float(dot(dot(WP, h), x))
    second_term = float(dot(WB, h))
    third_term = float(dot(WC, x))
    
    return float(first_term+second_term+third_term)

def compute_hidden_probability(k, x_vec):

    s = float(WB[k] + dot(WP[:,k], x_vec))
    u = random.uniform()
    e = float(exp(s))
    return 1 if float(e/(1.0+e)) > u else 0

def compute_observed_probability(d, h_vec):

    s = float(WC[d] + dot(WP[d], h_vec))
    u = random.uniform()
    e = float(exp(s))
    return 1 if float(e/(1.0+e)) > u else 0

def draw_figure(samples):

    figure = plt.figure()
    gs = gspec.GridSpec(10,10)
    img = [plt.subplot(gs[i]) for i in range(100)]

    plt.setp([a.get_xticklabels() for a in img], visible=False)
    plt.setp([a.get_yticklabels() for a in img], visible=False)

    for i in range(100):
        print "show " + str(i)
        img[i].imshow(samples[i].reshape(28, 28), cmap = cm.Greys_r)
    plt.show()   

def draw_energy_graph(E):

    #draw plot comparing energy levels for five 
    figure = plt.figure()
    plot = plt.plot(E)
    plt.legend(plot, ["sample #" + str(i) for i in range(5)])
    plt.ylabel("ENERGY") 
    plt.xlabel("Iterations")
    plt.title("Energy/Iterations")
    plt.show()

def GibbsSample():
    print "Running Gibbs Sampling..." 
    h = zeros([S+1, K])
    x = zeros([S+1, D])
    global e_count
    global last_sample
    global E
    sample = []  
    for k in xrange(0, K):
        h[0][k] = random.randint(0, 2)
    for s in xrange(1, S+1):
        for d in xrange(0, D):
            x[s][d] = compute_observed_probability(d, h[s-1])    
        for k in xrange(0, K):
            h[s][k] = compute_hidden_probability(k, x[s])
        if flag ==1 and s%5==0:  #Collect every 5th sample for Question 2a    
            sample.append(x[s])
        if flag ==2 and e_count < 5: #Store energy for first five chains
            E[e_count][s-1] = -energy(x[s], h[s])
    if flag==2:
        last_sample.append(x[-1])
        e_count+=1
    if flag == 1:
        draw_figure(sample)

def main():

    global K
    global S
    global D
    global E
    global flag
    K = 100
    S = 500
    D = 784
    E = zeros([5, S])    

    load_WP("Models/MNISTWP.txt")
    load_WB("Models/MNISTWB.txt")
    load_WC("Models/MNISTWC.txt")
    print "Question - 2a"
    
    flag = 1 #for 2a
    GibbsSample()
    
    print "Question - 2b"
    flag = 2 #for 2b
    for t in xrange(int(sys.argv[1])):
        print "Chain : " + str(t)
        GibbsSample()
    
    draw_figure(last_sample)
    E = E.T
    draw_energy_graph(E)
   
if __name__ == "__main__":

    main()
