#Author: Ashish Jain
#How to Run Code? python sumProduct.py <arg1> <arg2>
#arg 1 - path of test image file, arg2 - test word corresponding to that test file

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

correct_char = 0
total_char = 0

likelihood = 1.0

def load_FP(fname):
    #loading feature parameters into a dictionary
    global FP
    lines = open(fname, "r").readlines()
    l = []
    for i in xrange(0, len(lines)):
        FP[i] = map(lambda x: float(x), lines[i].strip().split())

def load_TP(fname):
    #loading transition parameter
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

     print "Node Potential: " + str(node_potential) 
     return node_potential

def compute_clique_potential(fname, word):

     # computing clique potential corresponding to each of the clique in markov network
     global clique_potential
     global size
        
     size = len(word)
 
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
                #normalizing each value in belief table                
                pairwise_marginal[i][ch1][ch2] = exp(beliefs[i][ch1][ch2])/normalizer

        for ch1 in ['t', 'a', 'h']:
            for ch2 in ['t', 'a', 'h']:
                print ch1 + " : " + ch2 + " " + str(pairwise_marginal[i][char_ordering[ch1]][char_ordering[ch2]])
       

        #adding up parisewise marginal probability along a row to compute marginal probability
        for j in xrange(10):
        
            marginal_distribution[i][j] = sum(pairwise_marginal[i][j])
            if i==l-1: 
                marginal_distribution[i+1][j] = sum(pairwise_marginal[i,:,j])
   
   for i in xrange(l+1):
        for j in char_ordering.keys(): 
        
                print str(j) + " " + str(marginal_distribution[i][char_ordering[j]]) + " ",
        print

def predict_character(correct_word):

    global correct_char
    global total_char
    #using marginal probability to predict character for a given state
    predicted_word = ""
    for i in xrange(0, len(marginal_distribution)):
        
        index =  argmax(array(marginal_distribution[i]).flatten())
        for char, order in char_ordering.items():
        
            if order==index:
                predicted_word+=char

    for j in xrange(0, len(predicted_word)):
        if predicted_word[j] == correct_word[j]:
            correct_char +=1 
    total_char += len(correct_word)
    print predicted_word
        

def average_loglikelihood(word):

    global likelihood 
    for i in xrange(0, len(word)):
        
        likelihood *= marginal_distribution[i][char_ordering[word[i]]]

def main_func(arg1, arg2):
    
    fname =  arg1
    word = arg2

    load_FP("model/feature-params.txt")
    load_TP("model/transition-params.txt")

    #Question 2.1
    print "Question 2.1"
    compute_clique_potential(fname, word)

    print
    print "Question 2.2"
    #Question 2.2
    sumproduct_message()
    
    print
    print "Question 2.3"
    #Question 2.3
    logbeliefs()
  
    print
    print "Question 2.4"
    #Question 2.4
    marginal_probability()

    print
    print "Question 2.5"
    #Question 2.5
    predict_character(word)    
   
    #Question 3.5
    average_loglikelihood(word)

def main():
    count =1
    for word in open("data/train_words.txt", "r"):
        main_func("data/train_img"+str(count)+".txt" , str(word.strip('\n')))
        count+=1
        if count == 51:
            break
    print correct_char
    print total_char
    print "average likelihod " +  str(log(likelihood)/50.0)
 
if __name__ == "__main__":
    #main() #uncomment this function if you want to see results for Question 3.5
    main_func(sys.argv[1], sys.argv[2])
