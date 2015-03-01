#Author: Ashish Jain

import os
import commands
import sys
import itertools
from numpy import dot
import math

char_ordering = {'e':0, 't':1, 'a':2, 'i':3, 'n':4, 'o':5, 's':6, 'h':7, 'r':8, 'd':9}

FP = {} #feature parameter
TP = [[0 for x in range(10)] for x in range(10)]  #transition parameter

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

def negative_energy(fname, word):
    
    node_potential = compute_node_potential(fname)
    energy_potential = 0.0      
    for i in xrange(0, len(word)-1):
        j = char_ordering[word[i]] #position j
        z = char_ordering[word[i+1]] #position j+1
        tp = TP[j][z]
        np = node_potential[i][j]
        energy_potential += tp+np
    energy_potential += node_potential[len(word)-1][char_ordering[word[-1]]]
    return energy_potential

def compute_log_partition(fname, word):

    len_sequence = len(word)
    vectors = [[key for key in char_ordering.keys()] for x in range(len_sequence)] 
    count=0
    partition_func=float(0.0)
    for combination in itertools.product(*vectors):
        count+=1
        word = ''.join(combination)
        energy = negative_energy(fname, word)
        partition_func+=math.exp(energy)
    print math.log(partition_func)

def best_labeling_sequence(fname, word):
    
    len_sequence = len(word)
    vectors = [[key for key in char_ordering.keys()] for x in range(len_sequence)]
    count=0
    partition_func=float(0.0)
    best_seq = ""
    max_energy = 0.0
    for combination in itertools.product(*vectors):
        count+=1
        word = ''.join(combination)
        
        energy = math.exp(negative_energy(fname, word))
        if energy > max_energy:
            best_seq = word
            max_energy = energy
        partition_func+=energy
    print best_seq
    print (max_energy/partition_func)

#def marginal_probability(fname, word):

    

def main():
 
    load_FP("model/feature-params.txt")
    load_TP("model/transition-params.txt")

    #Question-1.1
    #print compute_node_potential("data/test_img1.txt")

    #Question-1.2
    #negative_energy("data/test_img1.txt", "test")    
    #negative_energy("data/test_img2.txt", "hire")    
    #negative_energy("data/test_img3.txt", "rises")    
  
    #Question-1.3 
    #compute_log_partition("data/test_img1.txt", "test")
    #compute_log_partition("data/test_img2.txt", "hire")
    #compute_log_partition("data/test_img3.txt", "rises")

    #Question-1.4
    best_labeling_sequence("data/test_img1.txt", "test")
    best_labeling_sequence("data/test_img2.txt", "hire")
    best_labeling_sequence("data/test_img3.txt", "rises")

    #Question-1.5
    #marginal_probability("data/test_img1.txt", "test")

if __name__ == "__main__":
    main()
