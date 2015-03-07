#Author: Ashish Jain

import os
import sys
import itertools
from numpy import *
from math import *

char_ordering = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}

FP = {} #feature parameter
TP = [[0 for x in range(10)] for x in range(10)]  #transition parameter

clique_potential = None
beliefs = None
size = 0

forward_msg = {}
backward_msg = {}
marginal_distribution = None

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
     global size
        
     size = len(word)
 
     #clique_potential = [[[float (0.0) for x in range(10)] for x in range(10)] for x in range(len(word)-1)]
     clique_potential = zeros(shape=(len(word)-1, 10 ,10))

     node_potential = compute_node_potential(fname)
     for i in xrange(0, len(word)-1):
        #storing clique potential for each of the clique node
        if i == len(word)-2:          
            clique_potential[i] = matrix(node_potential[i]).T  + matrix(node_potential[i+1]) + TP
        else: 
            clique_potential[i] = matrix(node_potential[i]).T + TP
     for i in xrange(0, len(word)-1):
         for char1 in ['t', 'a', 'h']:
             for char2 in ['t', 'a', 'h']:
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
   backward_msg[size-1] = [0.0  for i in xrange(10)] 
   for i in xrange(size-2, 0, -1):
        key = str(i+1) + "->"+str(i) 
        potential = clique_potential[i]
        backward_msg[i] = []
        for j in xrange(0, 10):
            backward_msg[i].append(logsumexp(array(potential[j, :] + matrix(backward_msg[i+1])).flatten()))
        print key + ":" + str(backward_msg[i])

def logbeliefs():
    
    global beliefs
    
    beliefs = zeros(shape=(size-1, 10, 10))
 
    for i in xrange(size-1):
        
        if i == 0:
            beliefs[i] = clique_potential[i] +  matrix(backward_msg[i+1])
        elif i == size-2:
            beliefs[i] = clique_potential[i] + matrix(forward_msg[i+1]).T
        else:
            beliefs[i] = clique_potential[i] + matrix(backward_msg[i+1]) + matrix(forward_msg[i+1]).T
    #beliefs[0] = clique_potential[0] + matrix(backward_msg[1])
    #beliefs[1] = clique_potential[1] +  matrix(backward_msg[2]) + matrix(forward_msg[2]).T
    #beliefs[2] = clique_potential[2] + matrix(forward_msg[3]).T

    for i in xrange(0, len(beliefs)): 
        for ch1 in ['t', 'a']:
            for ch2 in ['t', 'a']:
                print ch1 + " : " + ch2 + " " + str(beliefs[i][char_ordering[ch1]][char_ordering[ch2]])

def marginal_probability():

   global marginal_distribution

   l = len(beliefs)
   pairwise_marginal = zeros(shape=(l, 10, 10))
   marginal_distribution = zeros(shape=(l+1, 10))
   for i in xrange(l):
        
        normalizer = 0.0          
        for ch1 in xrange(0, 10):
            for ch2 in xrange(0, 10):
                
                normalizer+=exp(beliefs[i][ch1][ch2])
        
        for ch1 in xrange(0,10):
            for ch2 in xrange(0,10):
                
                pairwise_marginal[i][ch1][ch2] = exp(beliefs[i][ch1][ch2])/normalizer

        for ch1 in ['t', 'a', 'h']:
            for ch2 in ['t', 'a', 'h']:
                print ch1 + " : " + ch2 + " " + str(pairwise_marginal[i][char_ordering[ch1]][char_ordering[ch2]])
       

        for j in xrange(10):
        
            marginal_distribution[i][j] = sum(pairwise_marginal[i][j])
            if i==l-1: 
                marginal_distribution[i+1][j] = sum(pairwise_marginal[i,:,j])
   
   for i in xrange(l+1):
        for j in char_ordering.keys(): 
        
                print str(j) + " " + str(marginal_distribution[i][char_ordering[j]]) + " ",
        print

def predict_character():

    predicted_word = ""
    for i in xrange(0, len(marginal_distribution)):
        
        index =  argmax(array(marginal_distribution[i]).flatten())
        for char, order in char_ordering.items():
        
            if order==index:
                predicted_word+=char
    print predicted_word
        
 
def main():

    load_FP("model/feature-params.txt")
    load_TP("model/transition-params.txt")

    #load_FP("/Users/ashishjain/Desktop/PGM/Assignments/pgm/Conditional Random Fields/Assignment2-PartA/Assignment2-PartA/model/feature-params.txt")
    #load_TP("/Users/ashishjain/Desktop/PGM/Assignments/pgm/Conditional Random Fields/Assignment2-PartA/Assignment2-PartA/model/transition-params.txt")

    #Question 2.1
    clique_potential("data/test_img1.txt", "test")
    #clique_potential("/Users/ashishjain/Desktop/PGM/Assignments/pgm/Conditional Random Fields/Assignment2-PartA/Assignment2-PartA/data/test_img5.txt", "strait")

    #Question 2.2
    sumproduct_message()
    
    #Question 2.3
    logbeliefs()
  
    #Question 2.4
    marginal_probability()

    #Question 2.5
    predict_character()    


if __name__ == "__main__":
    main()
