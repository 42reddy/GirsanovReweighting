#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 15:26:39 2024

This tests the one-dimensional polynomial potential 
implemented in the potential class

potential.D1(Polynomial)

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
sys.path.append("..")  

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
 
my_param = [0, 0, 8, 0.3, -6, 0, 1]
my_potential = D1.Polynomial(my_param)

print(" ")
print("class members: ")
print(dir(my_potential))
print(" ")
print("values of the parameters")
print("a: ",  my_potential.a)
print("c1: ",  my_potential.c1)
print("c2: ",  my_potential.c2)
print("c3: ",  my_potential.c3)
print("c4: ",  my_potential.c4)
print("c5: ",  my_potential.c5)
print("c6: ",  my_potential.c6)
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
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-2,2, 3):
        # set parameters
        this_param = [a, 0, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 2) / 4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='a={:.2f}'.format(a))
        
    plt.grid()
    plt.ylim(-10,20)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter a")
    plt.legend()
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c1
    plt.figure(figsize=(12, 6)) 
    for c1 in np.linspace(-3,3, 7):
        # set parameters
        this_param = [0, c1, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c1 + 3) / 6)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c1={:.2f}'.format(c1))
        
    plt.grid()
    plt.ylim(-12,12)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c1")
    plt.legend()    
    plt.show()
    plt.close()
    
    
    #----------------------
    # vary parameter c2
    plt.figure(figsize=(12, 6)) 
    for c2 in np.linspace(-2,10, 7):
        # set parameters
        this_param = [0, 0, c2, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c2 + 2) /12)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c2={:.2f}'.format(c2))
        
    plt.grid()
    plt.ylim(-30,12)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c2")
    plt.legend()  
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c3
    plt.figure(figsize=(12, 6)) 
    for c3 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, c3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c3 + 2) /4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c3={:.2f}'.format(c3))
        
    plt.grid()
    plt.ylim(-15,15)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c3")
    plt.legend()   
    plt.show()
    plt.close()

    #----------------------
    # vary parameter c4
    plt.figure(figsize=(12, 6)) 
    for c4 in np.linspace(-8,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, c4, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c4 + 8) /10)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c4={:.2f}'.format(c4))
        
    plt.grid()
    plt.ylim(-40,15)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c4")
    plt.legend()   
    plt.show()
    plt.close()

    #----------------------
    # vary parameter c5
    plt.figure(figsize=(12, 6)) 
    for c5 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, c5, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c5 + 2) /4)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c5={:.2f}'.format(c5))
        
    plt.grid()
    plt.ylim(-40,15)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c5")
    plt.legend()
    plt.show()
    plt.close()

    #----------------------
    # vary parameter c6
    plt.figure(figsize=(12, 6)) 
    for c6 in np.linspace(-0.5,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, 0, c6]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c6 + 0.5) /2.5)  # Normalize a to be in [0, 1]
        plt.plot(x, this_potential.potential(x) , color=color, label='c6={:.2f}'.format(c6))
        
    plt.grid()
    plt.ylim(-100,100)
    plt.xlabel("x")
    plt.ylabel("V(x)") 
    plt.title("Vary parameter c6")
    plt.legend()
    plt.show()
    plt.close()

# --------------------------------------------------------    
#  Plot force for various parameter values
if test_F == True: 
    print("---------------------------------")
    print(" testing the force   ")  
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-2,2, 3):
        # set parameters
        this_param = [a, 0, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 2) / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='a={:.2f}'.format(a))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-10,20)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter a")
    plt.legend()
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c1
    plt.figure(figsize=(12, 6)) 
    for c1 in np.linspace(-3,3, 7):
        # set parameters
        this_param = [0, c1, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c1 + 3) / 6)  # Normalize a to be in [0, 1]
  
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c1={:.2f}'.format(c1))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
       
    plt.grid()
    plt.ylim(-12,12)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c1")
    plt.legend()    
    plt.show()
    plt.close()   
    
    #----------------------
    # vary parameter c2
    plt.figure(figsize=(12, 6)) 
    for c2 in np.linspace(-2,10, 7):
        # set parameters
        this_param = [0, 0, c2, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c2 + 2) /12)  # Normalize a to be in [0, 1]

        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c2={:.2f}'.format(c2))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-30,12)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c2")
    plt.legend()  
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c3
    plt.figure(figsize=(12, 6)) 
    for c3 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, c3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c3 + 2) /4)  # Normalize a to be in [0, 1]
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c3={:.2f}'.format(c3))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)

    plt.grid()
    plt.ylim(-15,15)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c3")
    plt.legend()   
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c4
    plt.figure(figsize=(12, 6)) 
    for c4 in np.linspace(-8,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, c4, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c4 + 8) /10)  # Normalize a to be in [0, 1]
        
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c4={:.2f}'.format(c4))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-40,15)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c4")
    plt.legend()   
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c5
    plt.figure(figsize=(12, 6)) 
    for c5 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, c5, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c5 + 2) /4)  # Normalize a to be in [0, 1]
         
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c5={:.2f}'.format(c5))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
       
    plt.grid()
    plt.ylim(-40,15)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c5")
    plt.legend()
    plt.show()
    plt.close()

    #----------------------
    # vary parameter c6
    plt.figure(figsize=(12, 6)) 
    for c6 in np.linspace(-0.5,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, 0, c6]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c6 + 0.5) /2.5)  # Normalize a to be in [0, 1]
          
        # plot analytical force 
        plt.plot(x,  this_potential.force_ana(x)[0,:], color=color, label='c6={:.2f}'.format(c6))
        # plot numericaal  force 
        plt.plot(x, this_potential.force_num(x, h)[0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-100,100)
    plt.xlabel("x")
    plt.ylabel("F(x)") 
    plt.title("Vary parameter c6")
    plt.legend()      
    plt.show()
    plt.close()
    
# --------------------------------------------------------    
#  Plot Hessian for various parameter values
if test_H == True: 
    print("---------------------------------")
    print(" testing the Hessian")  
    
    #----------------------
    # vary parameter a
    plt.figure(figsize=(12, 6)) 
    for a in np.linspace(-2,2, 3):
        # set parameters
        this_param = [a, 0, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (a + 2) / 4)  # Normalize a to be in [0, 1]
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='a={:.2f}'.format(a))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-30,20)
    plt.xlabel("x")
    plt.ylabel("H(x)")  
    plt.title("Vary parameter a")
    plt.legend()
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c1
    plt.figure(figsize=(12, 6)) 
    for c1 in np.linspace(-3,3, 7):
        # set parameters
        this_param = [0, c1, 8, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c1 + 3) / 6)  # Normalize a to be in [0, 1]
  
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c1={:.2f}'.format(c1))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
      
    plt.grid()
    plt.ylim(-30,30)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c1")
    plt.legend()    
    plt.show()
    plt.close()   
    
    #----------------------
    # vary parameter c2
    plt.figure(figsize=(12, 6)) 
    for c2 in np.linspace(-2,10, 7):
        # set parameters
        this_param = [0, 0, c2, 0.3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c2 + 2) /12)  # Normalize a to be in [0, 1]

        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c2={:.2f}'.format(c2))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
       
    plt.grid()
    plt.ylim(-50,20)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c2")
    plt.legend()  
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c3
    plt.figure(figsize=(12, 6)) 
    for c3 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, c3, -6, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c3 + 2) /4)  # Normalize a to be in [0, 1]
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c3={:.2f}'.format(c3))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)

    plt.grid()
    plt.ylim(-50,20)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c3")
    plt.legend()   
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c4
    plt.figure(figsize=(12, 6)) 
    for c4 in np.linspace(-8,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, c4, 0, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c4 + 8) /10)  # Normalize a to be in [0, 1]
        
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c4={:.2f}'.format(c4))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
        
    plt.grid()
    plt.ylim(-70,50)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c4")
    plt.legend()   
    plt.show()
    plt.close()
    
    #----------------------
    # vary parameter c5
    plt.figure(figsize=(12, 6)) 
    for c5 in np.linspace(-2,2, 5):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, c5, 1]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c5 + 2) /4)  # Normalize a to be in [0, 1]
         
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c5={:.2f}'.format(c5))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
      
    plt.grid()
    plt.ylim(-150,20)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c5")
    plt.legend()
    plt.show()
    plt.close()

    #----------------------
    # vary parameter c6
    plt.figure(figsize=(12, 6)) 
    for c6 in np.linspace(-0.5,2, 6):
        # set parameters
        this_param = [0, 0, 8, 0.3, -6, 0, c6]
        # generate instance of the potential class
        this_potential = D1.Polynomial(this_param)
        
        # plot
        color = plt.cm.viridis( (c6 + 0.5) /2.5)  # Normalize a to be in [0, 1]
          
        # plot analytical Hessian 
        plt.plot(x,  this_potential.hessian_ana(x)[0,0,:], color=color, label='c6={:.2f}'.format(c6))
        # plot numericaal  Hessian 
        plt.plot(x, this_potential.hessian_num(x, h)[0,0,:], color=color, marker='o', linestyle='None', markersize=3)
       
    plt.grid()
    plt.ylim(-100,100)
    plt.xlabel("x")
    plt.ylabel("H(x)") 
    plt.title("Vary parameter c6")
    plt.legend()        
    plt.show()
    plt.close()  
    
  
# Test whether we can find minima and maxima of the potential
if test_TS == True: 
  print("---------------------------------")
  print(" testing the transition state finder")
  

# initialize potential
my_param = [0, 0, 8, 0.3, -6, 0, 1]  
my_potential = D1.Polynomial(my_param)   
  
# plot potential
plt.figure(figsize=(12, 6)) 
color = plt.cm.viridis(0) 
plt.plot(x, my_potential.potential(x) , color=color, label="polynomial potential")
    
plt.grid()
plt.ylim(-5,10)
plt.xlabel("x")
plt.ylabel("V(x)") 
plt.title("Transition states")
plt.legend()    
    
  
# Numerical solution
TS1_num = my_potential.TS(-1, 0)
TS2_num = my_potential.TS(0, 1)

TS_data = [
    ["x", TS1_num, TS2_num],
    ["Energy", my_potential.potential(TS1_num), my_potential.potential(TS2_num)],
    ["Force", my_potential.force_ana(TS1_num),my_potential.force_ana(TS2_num)],
    ["Hessian", my_potential.hessian_ana(TS1_num), my_potential.hessian_ana(TS2_num)],
]  
  
headers=["", "TS1", "TS2"]
table = tabulate(TS_data, headers=headers, tablefmt="grid")
print(table)

plt.plot(TS1_num, my_potential.potential(TS1_num) , color="red", marker="o")
plt.plot(TS2_num, my_potential.potential(TS2_num) , color="red", marker="o")
plt.show()
plt.close()    

    