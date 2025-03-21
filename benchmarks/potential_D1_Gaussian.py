#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:56:26 2024

This tests the one-dimensional logistic potential 
implemented in the potential class

potential.D1(Logistic)

@author: schaefej51
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
import os

# Get the absolute path of the parent directory
parent_dir = os.path.abspath("..")

# Append the parent directory to sys.path if it is not already included
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

print("---------------------------------------")
print("System path:")
print(sys.path)    

import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------
#   D1
#-----------------------------------------
# local packages and modules
from potential import D1

test_V = True
test_F = True
test_H = True
#test_TS = True



print("---------------------------------------")
print(" testing the class initialization   ")
 
my_param = [1, 1, 1]
my_potential = D1.Gaussian(my_param)

print(" ")
print("class members: ")
print(dir(my_potential))
print(" ")
print("values of the parameters")
print("k: ",  my_potential.k)
print("mu: ",  my_potential.mu)
print("sigma: ",  my_potential.sigma)
print(" ")
x = np.array([1, 2, 3, 4])

print("potential at x = ", x)
print(my_potential.potential(x)) 



# --------------------------------------------------------    
#  Plot potential for various parameter values
if test_V == True: 
    print("---------------------------------")
    print(" testing the potential   ")
    

    # set x-axis
    x = np.linspace(-5, 5, 501)
    
    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize k to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='k={:.2f}'.format(k))
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k, mu=0, sigma=a")
    plt.legend(loc='right')
    
    #----------------------
    # vary parameter mu
    plt.figure(figsize=(12, 6)) 
    for mu in np.linspace(-5,5, 11):
        # set parameters
        this_param = [20, mu, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (mu + 5) / 10)  # Normalize mu to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='mu={:.2f}'.format(mu))
     
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter mu, k=20, a=1")
    plt.legend(loc='right')

    #----------------------
    # vary parameter sigmaa
    plt.figure(figsize=(12, 6)) 
    for sigma in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: #to avoid division by zero
        # set parameters
        this_param = [20, 0, sigma]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (sigma + 5) / 10)  # Normalize sigmaa to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='sigma={:.2f}'.format(sigma))
     
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter sigma, k=20, mu=0")
    plt.legend(loc='right')

# --------------------------------------------------------    
#  Plot force for various parameter values
if test_F == True: 
    print("---------------------------------")
    print(" testing the force   ")

    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001
    
    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize k to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='k={:.2f}'.format(k))
        # plot numericaal  force 
        #plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=2)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter k, mu=0, sigma=1")
    plt.legend(loc='right')

    #----------------------
    # vary parameter mu
    plt.figure(figsize=(12, 6)) 
    for mu in np.linspace(-5,5, 11):
        # set parameters
        this_param = [20, mu, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (mu + 5) / 10)  # Normalize mu to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='mu={:.2f}'.format(mu))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=2)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter mu, k=20, sigma=1")
    plt.legend(loc='right')

    #----------------------
    # vary parameter sigma
    plt.figure(figsize=(12, 6)) 
    for sigma in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: #to avoid division by zero
        # set parameters
        this_param = [20, 0, sigma]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (sigma + 5) / 10)  # Normalize sigma to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='sigma={:.2f}'.format(sigma))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter sigma, k=20, mu=0")
    plt.legend()


# --------------------------------------------------------    
#  Plot Hessian
    
if test_H == True: 
    print("---------------------------------")
    print(" testing the Hessian   ")    

    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001    

    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize k to be in [0, 1]
        
        
        # plot analytical hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='k={:.2f}'.format(k))
        # plot numericaal  hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=2)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter k, mu=0, sigma=1")
    plt.legend(loc='right')

    #----------------------
    # vary parameter mu
    plt.figure(figsize=(12, 6)) 
    for mu in np.linspace(-5,5, 11):
        # set parameters
        this_param = [20, mu, 1]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (mu + 5) / 10)  # Normalize mu to be in [0, 1]
        
        
        # plot analytical hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='mu={:.2f}'.format(mu))
        # plot numericaal  hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=2)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter mu, k=20, sigma=1")
    plt.legend(loc='right')

    #----------------------
    # vary parameter sigma
    plt.figure(figsize=(12, 6)) 
    for sigma in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]: #to avoid division by zero
        # set parameters
        this_param = [20, 0, sigma]
        # generate instance of the potential class
        this_potential = D1.Gaussian(this_param)
        
        # plot
        color = plt.cm.viridis( (sigma + 5) / 10)  # Normalize sigma to be in [0, 1]
        
        
        # plot analytical hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='a={:.2f}'.format(sigma))
        # plot numericaal  hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-10,10)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter sigma, k=20, mu=0")
    plt.legend() 