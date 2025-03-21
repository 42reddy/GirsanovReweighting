#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 15:28:49 2024

@author: bettina
"""
#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np
import scipy.constants as const

# unnormalized Boltzmann factor
def boltzmann_factor(potential, x, T):
    """
    Calculate the unnormalized Boltzmann factor 
    
    The unnormalized Boltzmann factor is given by:
    p(x) = exp(- V(x)  / (R * T))
    
    
    Parameters:
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'potential' method that calculates the potential at a given position.
        - x (float): position 
        - T (float): temperature in units of K
          
    Returns:
        float: The value of the unnormalized Boltzmann factor at the given position x.
    """    
    # get natural constants in the appropriate units    
    R = const.R * 0.001
    
    return np.exp(- potential.potential(x)  / (T * R))
 
# velocity density for one-dimensional systems
def velocity_density_D1(v, m, T):
    """
    Calculate the one-dimensional velocity density as 
    
    The unnormalized Boltzmann factor is given by:
    p(v) = sqrt(m /(2 * pi * R * T) ) * exp(- v * v * m  / (2 * R * T))
    
    
    Parameters:
        - v (float): velocity
        - m (float): mass        
        - T (float): temperature in units of K
          
    Returns:
        float: The value of the one-dimensional velocity densityat the given velocity v.
    """    
    # get natural constants in the appropriate units    
    R = const.R * 0.001
    
    return np.sqrt(m /(2 * const.pi * R * T) ) * np.exp(- ( v * v * m )  / (2 * R * T))




    # # partition function
    # def partition_function(self, T, limits=None):
    #     """
    #     Calculate the partition function for the 1-dimensional Bolhuis potential 
        
    #     The partition function is given by:
    #     Q = \int_{-\infty}^{\infty} p(x) dx
        
    #     In practice, the integration bounds are set to specific values, specified in the variable limits.
    #     Integration is carried out by scipy.integrate
        
    #     Parameters:
    #         - T (float): temperatur in units of K
    #         - limits (list, optional): limits of he integrations. Defaults have been set at the initialization of the class 
                
    #     Returns:
    #         list: The value of the partition function and an error estimate [Q, Q_error] 
        
    #     """        
        
    #     try:         
    #         # no limits are passed, the class members x_low and x_high are available
    #         if limits==None and hasattr(self, 'x_low') and hasattr(self, 'x_high'):
    #             # integrate the unnormalized Boltzmann factor within the specified limits
    #             Q, Q_error = integrate.quad(self.boltzmann_factor, self.x_low, self.x_high,  args=(T))
    #             # return Q and the error estimates of the integation 
    #             return [Q, Q_error]
            
    #         # limits are passed, format is correct, ignore the class members x_low and x_high
    #         elif limits!=None and isinstance(limits, list) and len(limits) == 2:
    #             # integrate the unnormalized Boltzmann factor within the specified limits
    #             Q, Q_error = integrate.quad(self.boltzmann_factor, limits[0], limits[1],  args=(T))
    #             # return Q and the error estimates of the integation 
    #             return [Q, Q_error]
            
    #         # no limits are passed, but the class members x_low and x_high are missing
    #         elif limits==None and (not hasattr(self, 'x_low') or not hasattr(self, 'x_high') ):     
    #             raise ValueError("Default limits of the integration have not been implemented. Pass limits as an argument")
            
    #         # limit are passed, format is wrong, raise error
    #         elif limits!=None and (not isinstance(limits, list) or not len(limits) == 2):
    #             raise ValueError("Input 'limits' is not a list with two elements.")
            
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         return False




# #  Plot Boltzmann density for various temperatures
# if test_p == True: 
#     print("---------------------------------")
#     print(" testing the Boltzmann density   ")
#     # set x-axis
#     x = np.linspace(0, 5, 501)

#     # set parameters of the potential
#     this_param = [2, 1, 5, 2, 1, 10]
    
#     # generate instance of the potential class
#     this_potential = D1.Bolhuis(this_param)
    
#     # plot the potential
#     plt.figure(figsize=(12, 6)) 
#     plt.plot(x, this_potential.potential(x), color='blue', label='V(x)')

#     plt.ylim(0,20)
#     plt.xlabel("x")
#     plt.ylabel("V(x)") 
#     plt.title("Force for various values of alpha, line: analytical, dots: numerical")
#     plt.legend() 

#     # check error handling in the function partition function
#     print("no limits passed:")
#     print("Q = ", this_potential.partition_function(300))
#     limits = 3.0
#     print("limits is a float:")
#     print("Q = ", this_potential.partition_function(300, limits))
#     limits = [1,2,3]
#     print("limits is a an array with wrong length:")    
#     print("Q = ", this_potential.partition_function(300, limits))    

#     # plot partition function as a function of temperature
#     T_min = 200
#     T_max = 500 
#     T_list = np.linspace(T_min, T_max, 12)
#     Q_list = np.zeros( (12,2) )
#     N_list = np.zeros(12)

#     for i, T in enumerate(T_list): 
#         Q_list[i]  =  this_potential.partition_function(T_list[i])

#     plt.figure(figsize=(12, 6)) 
#     plt.plot(T_list, Q_list[:,0])
    
#     plt.xlabel("T")
#     plt.ylabel("Q(T)") 
#     plt.title("Partition functions as a function of temperature")
    
#     # plot Boltzmann densits for various temperatures
#     plt.figure(figsize=(12, 6)) 
#     for i, T in enumerate(T_list): 
#         color = plt.cm.coolwarm((T-T_min) / (T_max-T_min) )  # Normalize a to be in [0, 1]
    
#         # calculate partition function 
#         Q =  this_potential.partition_function(T)
    
#         # plot Boltzmann distrobution  
#         plt.plot(x, this_potential.boltzmann_factor(x, T)/Q[0], color=color, label='T={:.0f}'.format(T))
#         plt.xlabel("x")
#         plt.ylabel("p(x)") 
#         plt.title("Normalized Boltzmann density for various temperatures")
        
   
#     # check wethere thes Boltzmann densities are indeed normalized
#     # this is an indirect check whether Q is correct
#     # x grid needs to be large enough to cover most of p(x) left and right of the maximum
#     x_grid = np.linspace(-1, 5, 601)
#     # grid spacing 
#     dx = 6 / 600

#     for i, T in enumerate(T_list): 
#         # calcualate the normalization constant 
#         # using the Riemann integral on the grid x_grid
#         N_list[i] = np.sum( this_potential.boltzmann_factor(x_grid, T) ) * (1 / Q_list[i,0])* dx
 
#     plt.figure(figsize=(12, 6)) 
#     plt.plot(T_list, N_list)
    
#     plt.xlabel("T")
#     plt.ylabel("N") 
#     plt.title("Norm of the normalized Boltzmann density for various temperatures")
