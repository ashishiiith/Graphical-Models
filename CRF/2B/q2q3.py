#Author: Ashish Jain
#How to Run Code? python sumProduct.py <arg1> <arg2>
#arg 1 - path of test image file, arg2 - test word corresponding to that test file

import os
import sys
import time
import itertools
from numpy import *
from math import *
import scipy
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_l_bfgs_b

char_ordering = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}
y_label = {}

FP  = zeros( (10, 321) )
TP = zeros( (10, 10) )

node_potential =  None
clique_potential = None
beliefs = None
size = 0
n=10
feat_size = 321
num_words = None

forward_msg = {}
backward_msg = {}
pairwise_marginal = None
marginal_distribution = None
xij = None

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

     global node_potential
     global xij
     node_potential = []
     xij = []
     for vector in open(fname, "r"):
        feature_vec = map(lambda x:float(x), vector.split())
        vec = []
        for i in xrange(0, 10):
            vec.append(dot(FP[i], feature_vec))   #dot product of feature vector and learned feature parameter from model  
        node_potential.append(vec)
        xij.append(feature_vec) 
     #print "Node Potential: " + str(node_potential) 
     return node_potential

def compute_clique_potential(fname, word):

     # computing clique potential corresponding to each of the clique in markov network
     global clique_potential
     global size
        
     size = len(word)
 
     clique_potential = zeros(shape=(len(word)-1, 10 ,10))

     for i in xrange(0, len(word)-1):
        #storing clique potential for each of the clique node
        if i == len(word)-2:          
            clique_potential[i] = matrix(node_potential[i]).T  + matrix(node_potential[i+1]) + TP
        else: 
            clique_potential[i] = matrix(node_potential[i]).T + TP
     #for i in xrange(0, len(word)-1):
     #    for char1 in ['t', 'a', 'h']:
     #        for char2 in ['t', 'a', 'h']:
     #            print str(clique_potential[i][char_ordering[char1]][char_ordering[char2]]) + " ",
     #        print
     #    print 

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
        #print key + ":" + str(forward_msg[i+2])
   
   '''Implementing backward message passing
   '''
   backward_msg[size-1] = [0.0  for i in xrange(10)] 
   for i in xrange(size-2, 0, -1):
        key = str(i+1) + "->"+str(i) 
        potential = clique_potential[i]
        backward_msg[i] = []
        for j in xrange(0, 10):
            backward_msg[i].append(logsumexp(array(potential[j, :] + matrix(backward_msg[i+1])).flatten()))
        #print key + ":" + str(backward_msg[i])

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

    #for i in xrange(0, len(beliefs)): 
    #    for ch1 in ['t', 'a']:
    #        for ch2 in ['t', 'a']:
    #            print ch1 + " : " + ch2 + " " + str(beliefs[i][char_ordering[ch1]][char_ordering[ch2]])

def marginal_probability():

   global marginal_distribution
   global pairwise_marginal

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

        #for ch1 in ['t', 'a', 'h']:
        #    for ch2 in ['t', 'a', 'h']:
        #        print ch1 + " : " + ch2 + " " + str(pairwise_marginal[i][char_ordering[ch1]][char_ordering[ch2]])   

        #adding up parisewise marginal probability along a row to compute marginal probability
        for j in xrange(10):
        
            marginal_distribution[i][j] = sum(pairwise_marginal[i][j])
            if i==l-1: 
                marginal_distribution[i+1][j] = sum(pairwise_marginal[i,:,j])
   
   #for i in xrange(l+1):
   #     for j in char_ordering.keys(): 
        
   #             print str(j) + " " + str(marginal_distribution[i][char_ordering[j]]) + " ",
   #     print

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
    #print predicted_word
        

def average_loglikelihood(word):

    global likelihood 
    for i in xrange(0, len(word)):
        
        likelihood *= marginal_distribution[i][char_ordering[word[i]]]

def partition_function():

    l = []
    for i in xrange(0,10): 
        l.append(logsumexp(list(beliefs[0][i])))
 
    return logsumexp(l)

 
def loglikelihood(word):
    
    #compute energy 
    energy = 0.0
    for i in xrange(0, len(word)-1):
        energy+=clique_potential[i][char_ordering[word[i]]][char_ordering[word[i+1]]]
    logZ = partition_function()
   
    return energy - logZ

def sumProduct(fname, word):
    
    compute_node_potential(fname)
    compute_clique_potential(fname, word)
    sumproduct_message()
    logbeliefs()
    marginal_probability()
    predict_character(word)

def load_weights(wgts):
   
    global TP
    global FP
    
    i = 0
    for c in range(10):
        for cprime in range(10):
            TP[c][cprime] = wgts[i]
            i += 1
    for c in range(10):
        for f in range(321):
            FP[c][f] = wgts[i]
            i += 1
 
def objective_function(weights):
 
    global TP
    global FP   
 
    TP = weights[0:n*n].reshape([n, n])
    FP = weights[n*n:].reshape([n, feat_size])
   
    #load_weights(init_weights) 
    count =1
    likelihood = 0.0
    for word in open("../2A/data/train_words.txt", "r"):
        sumProduct("../2A/data/train_img"+str(count)+".txt" , str(word.strip('\n')))
        likelihood += loglikelihood(str(word.strip('\n')))
        count+=1
        if count == num_words+1:
            break
    avg_likelihood = -likelihood/float(num_words)
    return avg_likelihood

def gradient_function(weights):

    global TP
    global FP
    
    TP = weights[0:n*n].reshape([n, n])
    FP = weights[n*n:].reshape([n, feat_size])

    gradient_feat = zeros([10, feat_size])
    gradient_trans = zeros([10, 10])
    
    count = 1
    for words in open("../2A/data/train_words.txt", "r"):
       
       word = str(words.strip('\n'))
       sumProduct("../2A/data/train_img"+str(count)+".txt" , word)
       
       #for transition distribution
       for i in xrange(0, size-1):
           label1 = char_ordering[word[i]]
           label2 = char_ordering[word[i+1]]
           gradient_trans[label1][label2] += 1
           for label1 in xrange(0, 10):
               for label2 in xrange(0, 10):       
                   gradient_trans[label1][label2] -= pairwise_marginal[i][label1][label2]
       #print "tansition gradient\n"
       #print gradient_trans    
       
       #for marginal distribution
       #print "len xij : " +  str(word) + " " + str( len(xij))
       for i in xrange(0, size):
           label = char_ordering[word[i]]
           for f in xrange(0, feat_size):
               gradient_feat[label][f]+=xij[i][f]
               for c in xrange(0, 10):
                   gradient_feat[c][f] -= marginal_distribution[i][c]*xij[i][f]
       count+=1
       if count == num_words+1:
           break
    gradient_feat = concatenate(gradient_feat, axis=1)
    gradient_trans = concatenate(gradient_trans, axis=1)
    print -concatenate([gradient_trans, gradient_feat], axis=1)/float(num_words)
    return -concatenate([gradient_trans, gradient_feat], axis=1)/float(num_words)


def output_result(result):
  
    fd = open("result", "a")
    for val in result:
        fd.write(str(val) + " ")
        fd.flush()
    fd.close()

def main():

    t0 = time.clock()
    global num_words
    wgts = open("result", "r").readline().split()
    load_weights(wgts)   
    count = 1
    for word in open("../2A/data/test_words.txt", "r"):
        sumProduct("../2A/data/test_img"+str(count)+".txt" , str(word.strip('\n')))
        count+=1
    print correct_char
    print total_char

if __name__ == "__main__":
    main() #uncomment this function if you want to see results for Question 3.5
