#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:43:45 2024

This tests the one-dimensional Prinz potential 
implemented in the potential class

potential.D1(Prinz)


@author: bettina
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
from tabulate import tabulate


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
 
my_potential = D1.Prinz()

print(" ")
print("class members: ")
print(dir(my_potential))
print(" ")
x = np.array([1, 2, 3, 4])

print("potential at x = ", x)
print(my_potential.potential(x)) 


# --------------------------------------------------------    
#  Set global parameters
# set x-axis
x = np.linspace(-1.5, 1.5, 151)
# set discretization interval
h = 0.001

# --------------------------------------------------------    
#  Plot potential for various parameter values
if test_V == True: 
    print("---------------------------------")
    print(" testing the potential   ")
    
    plt.figure(figsize=(12, 6)) 
        
    # plot
    color = plt.cm.viridis(0)  # Normalize a to be in [0, 1]
    plt.plot(x, my_potential.potential(x) , color=color)
        
    plt.grid()
    plt.ylim(0,6)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Prinz potential")
    plt.show()
    plt.close()  

if test_F==True: 
    print("---------------------------------")
    print(" testing the force   ")
    
    plt.figure(figsize=(12, 6)) 
        
    # plot
    color = plt.cm.viridis(0)  # Normalize a to be in [0, 1]
    plt.plot(x, my_potential.force(x, h)[0,:] , color=color)
        
    plt.grid()
    plt.ylim(-30,30)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Prinz potential")    
    plt.show()
    plt.close()  

if test_H==True: 
    print("---------------------------------")
    print(" testing the force   ")
    
    plt.figure(figsize=(12, 6)) 
        
    # plot
    color = plt.cm.viridis(0)  # Normalize a to be in [0, 1]
    plt.plot(x, my_potential.hessian(x, h)[0,0,:] , color=color)
        
    plt.grid()
    plt.ylim(-550,250)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Prinz potential")        
    plt.show()
    plt.close()  
    
if test_TS==True:    
    print("---------------------------------")
    print(" testing the transition state finder")

      
    # plot potential
    plt.figure(figsize=(12, 6)) 
    color = plt.cm.viridis(0)  # Normalize a to be in [0, 1]
    plt.plot(x, my_potential.potential(x) , color=color)
        
    plt.grid()
    plt.ylim(0,6)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Prinz potential")
        
      
    # Numerical solution
    TS1_num = my_potential.TS(-.75, -0.25)
    TS2_num = my_potential.TS(-0.25, 0.25)
    TS3_num = my_potential.TS(0.25, 0.75)

    
    TS_data = [
        ["x", TS1_num, TS2_num, TS3_num],
        ["Energy", my_potential.potential(TS1_num), my_potential.potential(TS2_num), my_potential.potential(TS3_num)],
        ["Force", my_potential.force(TS1_num, h),my_potential.force(TS2_num, h), my_potential.force(TS3_num, h)],
        ["Hessian", my_potential.hessian(TS1_num, h), my_potential.hessian(TS2_num, h), my_potential.hessian(TS3_num, h)],
    ]  
      
    headers=["", "TS1", "TS2", "TS3"]
    table = tabulate(TS_data, headers=headers, tablefmt="grid")
    print(table)
    
    plt.plot(TS1_num, my_potential.potential(TS1_num) , color="red", marker="o")
    plt.plot(TS2_num, my_potential.potential(TS2_num) , color="red", marker="o")
    plt.plot(TS3_num, my_potential.potential(TS3_num) , color="red", marker="o")
    plt.show()
    plt.close()     