#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:37:03 2024

This tests the one-dimensional constant potential 
implemented in the potential class

potential.D1(Constant)

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
#test_TS = True

# --------------------------------------------------------    
#  Plot potential for various parameter values
if test_V == True: 
    print("---------------------------------")
    print(" testing the Potential   ")
    
    
    my_param = [5]
    my_potential = D1.Constant(my_param)
    
    print(" ")
    print("class members: ")
    print(dir(my_potential))
    print(" ")
    print("values of the parameters")
    print("d: ",  my_potential.d)
    print(" ")
    x = np.array([1, 3, 4])
    
    print("potential at x = ", x)
    print(my_potential.potential(x))    

    # set x-axis
    x = np.linspace(-5, 5, 501)
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for d in np.linspace(-5,5, 11):
        # set parameters
        this_param = [d]
        # generate instance of the potential class
        this_potential = D1.Constant(this_param)
        
        # plot
        color = plt.cm.viridis((d + 2) / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='d={:.2f}'.format(d))
        
    plt.ylim(-6,6)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter d")
    plt.legend()
    
# --------------------------------------------------------    
#  Compare force calculated analytically and numerically
if test_F == True: 
    print("---------------------------------")
    print(" testing the force   ")
    
    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001

    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(12, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
 
    # set parameters          
    this_param = [2]
    # generate instance of the potential class
    this_potential = D1.Constant(this_param)
    # colors
    color = plt.cm.viridis(2 / 4)  # Normalize a to be in [0, 1]
     
    # plot analytical force 
    plt.plot(x, this_potential.force_ana(x)[0,:], color=color, label='d={:.2f}'.format(d))
    # plot numericaal  force 
    plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
 
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Force, line: analytical, dots: numerical")
    plt.legend()    
    
# --------------------------------------------------------    
#  Compare force calculated analytically and numerically
if test_H == True: 
    print("---------------------------------")
    print(" testing the Hessian   ")
    
    # set x-axis
    x = np.linspace(-5, 5, 501)
    h = 0.001

    #----------------------
    # vary parameter alpha
    plt.figure(figsize=(12, 6)) 
    plt.axhline(y=0, color='black', linestyle='dotted', linewidth=1)
 
    # set parameters          
    this_param = [2]
    # generate instance of the potential class
    this_potential = D1.Constant(this_param)
    # colors
    color = plt.cm.viridis(2 / 4)  # Normalize a to be in [0, 1]
     
    # plot analytical hessian 
    plt.plot(x, this_potential.hessian_ana(x)[0,0,:], color=color, label='d={:.2f}'.format(d))
    # plot numericaal  heassian 
    plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
 
    plt.ylim(-20,20)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Hessian, line: analytical, dots: numerical")
    plt.legend()        