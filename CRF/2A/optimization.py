#Using fmin_bfgs optimizer from python scipy package
#Run the code: python optimization.py 
import os
import sys
import itertools
from numpy import *
from scipy.optimize import fmin_bfgs


def objective_f(x):
    #computing objective function    
    val = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
    return val

def gradient_f(x):
   #computing gradient for the given function
   derivative = zeros_like(x)
   derivative[0] = 2*(x[0]-1)+400*(x[0])*(x[0]**2 - x[1])
   derivative[1] = 200*(x[1]-x[0]**2)
   return derivative

def main():
 
    x0 = [1.3, 0.7] 
    xopt = fmin_bfgs(objective_f, x0, fprime=gradient_f, disp=True, retall=True)
    print xopt

if __name__ == "__main__":
    main()
