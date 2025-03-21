#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 07:31:14 2024

This tests the one-dimensional double-well potential 
implemented in the potential class

potential.D1(DoubleWell)

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
sys.path.append("..")  

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
test_TS = True


print("---------------------------------------")
print(" testing the class initialization   ")
 
my_param = [5, 2, 3]
my_potential = D1.DoubleWell(my_param)

print(" ")
print("class members: ")
print(dir(my_potential))
print(" ")
print("values of the parameters")
print("k: ",  my_potential.k)
print("a: ",  my_potential.a)
print("b: ",  my_potential.b)
print(" ")
x = np.array([1, 2, 3, 4])

print("potential at x = ", x)
print(my_potential.potential(x)) 


# --------------------------------------------------------    
#  Set gloabl parameters: 

# set x-axis
x = np.linspace(-5, 5, 501)
# h 
h = 0.001

# --------------------------------------------------------    
#  Plot potential for various parameter values
if test_V == True: 
    print("---------------------------------")
    print(" testing the potential   ")  
    
    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='k={:.2f}'.format(k))
        
    plt.grid()
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter k, a=0, b=2")
    plt.legend()
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-4,4, 3):
        # set parameters
        this_param = [3, a, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 5) / 10)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='a={:.2f}'.format(a))
     
    plt.grid()
    plt.ylim(-0,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter a, k=3, b=2")
    plt.legend()

    #----------------------
    # vary parameter b
    plt.figure(figsize=(12, 6)) 
    for b in np.linspace(-5,5, 11):
        # set parameters
        this_param = [3, 0, b]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (b + 5) / 10)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='b={:.2f}'.format(b))
     
    plt.grid()
    plt.ylim(-0,100)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter b, k=3, a=0")
    plt.legend()  
    
    
    
# --------------------------------------------------------    
#  Plot force for various parameter values
if test_F == True: 
    print("---------------------------------")
    print(" testing the force   ")

    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='k={:.2f}'.format(k))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter k, a=0, b=2")
    plt.legend()    
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-5,5,11):
        # set parameters
        this_param = [3, a, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='a={:.2f}'.format(a))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter a, k=3, b=2")
    plt.legend()       
    
    #----------------------
    # vary parameter b
    plt.figure(figsize=(12, 6)) 
    for b in np.linspace(-5,5,11):
        # set parameters
        this_param = [3, 0, b]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (b + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='b={:.2f}'.format(b))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter b, k=3, a=0")
    plt.legend()       
   
# --------------------------------------------------------    
#  Plot Hessian for various parameter values
if test_H == True: 
    print("---------------------------------")
    print(" testing the Hessian   ")
    
    #----------------------
    # vary parameter k
    plt.figure(figsize=(12, 6)) 
    for k in np.linspace(-5,5, 11):
        # set parameters
        this_param = [k, 0, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (k + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='k={:.2f}'.format(k))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter k, a=0, b=2")
    plt.legend()       
  
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-5,5, 11):
        # set parameters
        this_param = [3, a, 2]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='a={:.2f}'.format(a))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-30,10)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter a, k=3, b=2")
    plt.legend()       
   
    #----------------------
    # vary parameter b
    plt.figure(figsize=(12, 6)) 
    for b in np.linspace(-5,5, 11):
        # set parameters
        this_param = [3, 0, b]
        # generate instance of the potential class
        this_potential = D1.DoubleWell(this_param)
        
        # plot
        color = plt.cm.viridis( (b + 5) / 10)  # Normalize a to be in [0, 1]
        
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='b={:.2f}'.format(b))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
    
        
    plt.grid()
    plt.ylim(-80,80)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter b, k=3, a=0")
    plt.legend()   
    
# Test whether we can find minima and maxima of the potential
if test_TS == True: 
  print("---------------------------------")
  print(" testing the transition state finder")
  
  # keep set a=0, then the TS is at x=0
  my_param = [3,0,2]
  my_potential = D1.DoubleWell(my_param)    
  min_1 = -1
  min_2 = 1
  TS = 0
  
  print (" ")
  print ("Analytical solution ")
  print ("transition state:", TS )
  print ("energy:", my_potential.potential(TS) ) 
  print ("force: ",  my_potential.force_ana(TS) ) 
  print ("hessian: ", my_potential.hessian_ana(TS) ) 

  print (" ")
  print ("Numerical solution provided by class")
  TS_num = my_potential.TS(min_1, min_2)
  print ("transition state:", TS_num )
  print ("energy:", my_potential.potential(TS_num) ) 
  print ("force: ",  my_potential.force_ana(TS_num) ) 
  print ("hessian: ", my_potential.hessian_ana(TS_num) )   
    
    
    