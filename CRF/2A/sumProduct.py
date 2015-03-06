import os
import sys
import itertools
from numpy import *

char_ordering = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}

FP = {} #feature parameter
TP = [[0 for x in range(10)] for x in range(10)]  #transition parameter

clique_potential = None
beliefs = None

forward_msg = {}
backward_msg = {}

def load_FP(fname):

    global FP
    lines = open(fname, "r").readlines()
    l = []
    for i in xrange(0, len(lines)):
        FP[i] = map(lambda x: float(x), lines[i].strip().split())

def load_TP(fname):

    global TP
    y = 0
    for line in open(fname, "r"):
        x = 0
        tokens = line.strip().split()
        for token in tokens:
            TP[x][y] = float(token)
            x+=1
        y+=1

def compute_node_potential(fname):

     node_potential = []

     for vector in open(fname, "r"):
        feature_vec = map(lambda x:float(x), vector.split())
        vec = []
        for i in xrange(0, 10):
            vec.append(dot(FP[i], feature_vec))   #dot product of feature vector and learned feature parameter from model  
        node_potential.append(vec)

     #print node_potential 
     return node_potential

def clique_potential(fname, word):

     global clique_potential 
     #clique_potential = [[[float (0.0) for x in range(10)] for x in range(10)] for x in range(len(word)-1)]
     clique_potential = zeros(shape=(len(word)-1, 10 ,10))

     node_potential = compute_node_potential(fname)
     for i in xrange(0, len(word)-1):
        #storing clique potential for each of the clique node
        if i == len(word)-1:          
            clique_potential[i] = matrix(node_potential[i]).T  + matrix(node_potential[i+1]) + TP
        else: 
            clique_potential[i] = matrix(node_potential[i]).T + TP
        #print clique_potential[i]
     #print array(clique_potential[0][1]).flatten()[1]
     for i in xrange(0, len(word)-1):
         for char1 in ['t', 'a', 'h']:
             #print char_ordering[char1]
             for char2 in ['t', 'a', 'h']:
                 #print char_ordering[char2]
                 print str(clique_potential[i][char_ordering[char1]][char_ordering[char2]]) + " ",
             print
         print 

def logsumexp(vector):

    #print vector
    c = max(vector)
    vector = map(lambda x : math.exp(x-c), vector)
    return c + math.log( sum(vector) )

def sumproduct_message():

   global forward_msg
   global backward_msg

   potential = zeros(shape=(10,10))
   
   ''' Implementing forward message passing.
   '''
   forward_msg[1] = [0.0  for i in xrange(10)]
   for i in xrange(0, len(clique_potential)-1):
        key = str(i+1) + "->" + str(i+2)
        potential = clique_potential[i]
        forward_msg[i+2] = [] 
        
        for j in xrange(0, 10):
            forward_msg[i+2].append(logsumexp(array(potential[:,j]+matrix(forward_msg[i+1])).flatten()))
        print key + ":" + str(forward_msg[i+2])
   
   '''Implementing backward message passing
   '''
   backward_msg[3] = [0.0  for i in xrange(10)] 
   for i in xrange(2, 0, -1):
        key = str(i+1) + "->"+str(i) 
        potential = clique_potential[i]
        backward_msg[i] = []
        for j in xrange(0, 10):
            backward_msg[i].append(logsumexp(array(potential[j, :] + matrix(backward_msg[i+1])).flatten()))
        print key + ":" + str(backward_msg[i])

def logbeliefs():
    
    global beliefs
    
    beliefs = zeros(shape=(3, 10, 10))
 
    beliefs[0] = clique_potential[0] + matrix(backward_msg[1])
    beliefs[1] = clique_potential[1] +  matrix(backward_msg[2]) + matrix(forward_msg[2]).T
    beliefs[2] = clique_potential[2] + matrix(forward_msg[3]).T

    for i in xrange(0, len(beliefs)): 
        for ch1 in ['t', 'a']:
            for ch2 in ['t', 'a']:
                print ch1 + " : " + ch2 + " " + str(beliefs[i][char_ordering[ch1]][char_ordering[ch2]])

def marginal_porbability():



def main():

    load_FP("model/feature-params.txt")
    load_TP("model/transition-params.txt")

    #Question 2.1
    clique_potential("data/test_img1.txt", "test")

    #Question 2.2
    sumproduct_message()
    
    #Question 2.3
    logbeliefs()
  
    #Question 2.4
    marginal_probability()

if __name__ == "__main__":
    main()
