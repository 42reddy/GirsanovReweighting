#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 09:28:56 2024

@author: bettina

Next steps:
    
    #------------------------------------------  
    # package "utils"
    #------------------------------------------
    # module "rate_theory" 

    - implement Kramers
    - implement transition count across barrier
    - implement Peter's rate estimate

    #------------------------------------------
    # module "thermodynamics"

    - implement partition function
    - implement entropy

    #------------------------------------------------
    # create a new module "FP_discretized" 
    
    - rate_theory:: SqRA
    - within rate_theory, write a module rate_matrix for rates from rate_matrices
    - rateMatrix: SqRA_rate (via Berezhkovski, Szabo)    
    - rateMatrix: SqRA_rate_via_its (as inverse of ITS)    
    - rateMatrix: MSM_rate_via_its (as inverse of ITS)   

    #------------------------------------------------
    # create a new module "Markov_model" 

    #------------------------------------------
    # package "integrators" 
    #------------------------------------------
    # module "stochastic"
    - implement EM algorithm
    - implement integrators for biased potentials

    #------------------------------------------
    # package "system" 
    
    - make v optional at initialization, draw v from Maxwell-Boltzmann dist
    - add variables for bias force and random number
    - add variables for force -> more efficient implementation of Langevin integrators

    #------------------------------------------
    # package "potentials" 
    #------------------------------------------
    - D1: GaussianBias
    - D1: Morse potential
    - D1: Lennard Jones potential
    
    #------------------------------------------
    # documentation
    #------------------------------------------
    - update main README file
    - add README files to each package that explain the code structure of the package
    
    #------------------------------------------
    # cook book
    #------------------------------------------
    - Langevin simulation

    #------------------------------------------
    # application
    #------------------------------------------
    - Estimate rates from Kramers theory
    - Simulate Bolhuis potential for various values of alpha, 5 traj each, Langevin dynamics ABOBA
        - 1e5 steps
        - 1e6 steps
        - 1e7 steps
        - 1e8 steps (?)
    - Estimate rate via transition count
    - Estimate rate via Peter's estimate
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

# local packages and modules
from system import system
from potential import D1
from integrator import stochastic as sd
from utils import thermodynamics as thermo



#-----------------------------------------
#   S Y S T E M
#-----------------------------------------
m = 100.0
x = 0.0
v = 0.0 
T = 300.0
xi = 1.0
dt = 0.1
# initialize the system
system = system.D1(m, x, v, T, xi, dt)


print("-----------------------------------------------------------------------")
print(" Initialized system ")
print("-----------------------------------------------------------------------")
print("Mass: ", system.m, " u")
print("Position: ", system.x, " nm")
print("Velocity: ", system.v, " nm/ps")
print("Temperature: ", system.T, " K")
print("Collision frequency: ", system.xi, " 1/ps")
print("Time step: ", system.dt, " ps")


#-----------------------------------------
#   P O T E N T I A L 
#-----------------------------------------

# initialize one-dimensional Bolhuis potential
# the simulation potential
param = [2, 2, 5 , 1, 1, 4]
potential = D1.Bolhuis(param)
# the reference potential, just for plottin
param_ref = [2, 2, 5 , 1, 1, 0]
potential_ref = D1.Bolhuis(param_ref)

# set x-axis
x = np.linspace(-1, 5, 61)
# set bin width
dx = x[1]-x[0]

print("-----------------------------------------------------------------------")
print(" Initialized 1-dimensional Bolhuis potential ")
print("-----------------------------------------------------------------------")
print("a: ", potential.a)
print("b: ", potential.b)
print("c: ", potential.c)
print("k1: ", potential.k1)
print("k2: ", potential.k2)
print("alpha: ", potential.alpha)


#----------------------
# vary parameter alpha
plt.figure(figsize=(12, 6)) 

plt.plot(x, potential.potential(x) , color="darkblue", label='alpha={:.2f}'.format(potential.alpha))
plt.plot(x, potential_ref.potential(x) , color="darkblue", linestyle = "dashed", label='alpha={:.2f}'.format(potential_ref.alpha))
    
plt.ylim(0,50)
plt.xlabel("x in nm", fontsize=14)
plt.ylabel("V(x) in kJ/mol", fontsize=14) 
plt.title("The potential energy function", fontsize=16)
plt.legend(fontsize=16)

# delete reference potential and corresponding parameter list
del(param_ref)
del(potential_ref)


#-----------------------------------------
#   S I M U L A T I O N 
#-----------------------------------------
print("-----------------------------------------------------------------------")
print(" Langevin simulation ")
print("-----------------------------------------------------------------------")

# set random number seed to obtain reproducible random numbers
#np.random.seed(42)

# number of simulation step
n_steps = 1_000_000
# frequency at which position and velocities are reported
n_steps_out = 10
# frequency at which progress is written to the terminal
n_steps_report = np.floor(n_steps / 10)

# initialize position and velocity vector 
# to the current position and velocity
pos = np.array([system.x])
vel = np.array([system.x])

# loop over simulation steps
for k in range(n_steps):
    
    # report progress
    if k % n_steps_report == 0:
        print('Simulation progress: ', k/n_steps * 100, '%')
    
    # perform an integration step
    sd.ABO(system, potential)

    # save position and velocity to an array
    if k % n_steps_out == 0:
        pos = np.append(pos, system.x)
        vel = np.append(vel, system.v)

#-----------------------------------------
#   A N A L Y S I S
#-----------------------------------------
print("-----------------------------------------------------------------------")
print(" Analysis ")
print("-----------------------------------------------------------------------")

#--------------------------------
#   Time series
#--------------------------------
print("Plotting time series")
plt.figure(figsize=(12, 6)) 
plt.plot(pos)

plt.xlabel("time in ps", fontsize=14)
plt.ylabel("x in nm", fontsize=14) 
plt.title("Position time series", fontsize=16)
#plt.legend(fontsize=16)


#--------------------------------
#   Stationary density
#--------------------------------
print("Calculating and plotting stationary density")

# get a histogram of the sampled density
counts_x, bins_x = np.histogram(pos, bins=x, density=True)

# get analytical boltzmann density
boltzmann_density = thermo.boltzmann_factor(potential, x[0:-1] + ( dx /2 ), system.T)
# normalize boltzmann_density
boltzmann_density = boltzmann_density / (np.sum(boltzmann_density) * dx)

# plot stationary density
plt.figure(figsize=(12, 6)) 
plt.stairs(counts_x, bins_x, label="sampled density")
plt.stairs(boltzmann_density, bins_x, label="Boltzmann density")

plt.xlabel("x in nm", fontsize=14) 
plt.ylabel("p(x)", fontsize=14)
plt.title("stationary density", fontsize=16)
plt.legend(fontsize=16)


#--------------------------------
#   Velocity density
#--------------------------------
print("Calculating and plotting velocity density")

# get a histogram of the sampled density
counts_v, bins_v = np.histogram(vel, bins=100, density=True)

# get bin centers
dv = bins_v[1] - bins_v[0]
bin_centers = bins_v[0:-1] + ( dv /2 )

# get analytical velocity density
velocity_density = thermo.velocity_density_D1(bin_centers, system.m, system.T)
# normalize boltzmann_density
boltzmann_density = boltzmann_density / dv

# plot velocity density
plt.figure(figsize=(12, 6)) 
plt.stairs(counts_v, bins_v, label="sampled density")
plt.stairs(velocity_density, bins_v, label="velocity density")

plt.xlabel("v in nm/ps", fontsize=14) 
plt.ylabel("p(x)", fontsize=14)
plt.title("velocity density", fontsize=16)
plt.legend(fontsize=16)


#--------------------------------
#   Phase space density
#--------------------------------
print("Calculating and plotting phase space density")

phase_space_density = np.outer(counts_x, counts_v)

plt.figure(figsize=(10, 10)) 
plt.pcolormesh(phase_space_density,
            cmap='RdBu'
            )

print("-----------------------------------------------------------------------")
print("END ")
print("-----------------------------------------------------------------------")
