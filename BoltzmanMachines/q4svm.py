import os
import svmlight_write as sw
from numpy import *
import pandas as pd
import sys
WB = None
WP = None 
K = 784
n = None
z = None

def probability(x):

    e = exp(x)
    return e/(1+e)

global n
global z

n=60000
z = zeros([60000, K])

WB = loadtxt("WB.txt") #give WB parameter
WP = loadtxt("WP.txt") #give WP parameter
print WB.shape
print WP.shape
print "Reading data..."
data = pd.read_csv(sys.argv[1],sep=' ', dtype=int, header=None)
xn = data.values
print xn.shape
features = probability(dot(xn, WP) + WB)

print "Reading labels.."
labels = pd.read_csv(sys.argv[2], sep=' ', dtype=int, header=None).values

print features.shape
print "Writing svm format file..."
#sw.svmlight_write(labels, features, 'train.txt')


