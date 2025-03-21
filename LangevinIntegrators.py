#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:41:03 2023

@author: bettina
"""

#--------------------------------
#  I M P O R T S
#--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

#--------------------------------
#  P A R A M E T E R S
#--------------------------------
random_seed = 202309220948




# set the seed for the random number generator
random.seed(random_seed)

# genrate sequence of random numbers
eta_mu, eta_sigma = 0, 1.0 # mean and standard deviation
eta = np.random.normal(eta_mu, eta_sigma, 100000)

# print deviation of data from mean
print(abs(eta_mu - np.mean(eta)))
# print deviation of data from mean
print(abs(eta_sigma - np.std(eta, ddof=1)))
# test whethe the correct distribution is reproduced
count, bins, ignored = plt.hist(eta, 30, density=True)
plt.plot(bins, 1/(eta_sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - eta_mu)**2 / (2 * eta_sigma**2) ), linewidth=2, color='r')
plt.show()
#----------------------------------------
# H E R E 
# test for correlations in the data
#----------------------------------------


#-----------------------------------------------------------------------
#- write a function that tests the quality of random number sequences
#- copy function from Bolhuis double well potential
#- copy function from Bolhuis double well force
#- document potential and force
#- implement A, B, O
#- sample potential with ABO, determine step size, friction, T etc.
#