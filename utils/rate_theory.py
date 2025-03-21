#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 05:14:50 2024

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import sys
sys.path.append("..")  

import numpy as np
import scipy.constants as const

# local packages and modules
from potential import D1
#from system import system


def TST_D1(x_A, x_TS, T, m, h, potential): 
    """
    Calculate the rate constant using Eyring Transition State Theory (TST) for a unimolecular reaction.

    Parameters:
    - x_A (float): Position of reactant A.
    - x_TS (float): Position of the transition state (TS).
    - T (float): Temperature in Kelvin.
    - m (float): Mass of the reacting molecule.
    - h (float): Discretization interval for numerical Hessian
    - potential (object): Object representing the potential energy surface, with methods:
        - potential.potential(x): Calculate the potential energy at position x.
        - potential.hessian(x, h): Calculate the Hessian matrix at position x.
      potential.hessian(x, h) uses the analytical Hessian if implemtend, and the numerical Hessian otherwise   

    Returns:
    - k_AB (float): Rate constant for the reaction from A to B.

    The TST formula used is based on the assumption of a harmonic transition state and the 
    separation of reactant and transition state motions. The vibrational partition function 
    of the transition state (q_TS) is set to one, and the partition function of A (q_A) is 
    calculated based on the vibrational frequency and temperature. The energy barrier is 
    calculated as (E_b) as the difference in the potential at the transition state and at
    the reactant state.

    The rate constant (k_AB) is then computed using the Eyring TST
    
    k_AB = R*T / h *  (q_TS / q_A) * np.exp(- E_b / (R * T)) 

    The unit of k_AB is 1/ps

    Note: This function assumes a one-dimensional reaction coordinate.
    """
    
    # get natural constants in the right units    
    h = const.h * const.Avogadro * 0.001 *  1e12
    R = const.R * 0.001
    
    # get the energy barrier
    E_b = potential.potential(x_TS) - potential.potential(x_A)

    #get the force consant in the transition state
    k_A = potential.hessian(x_A, h)[0,0]
    
    # calculate the frequency
    nu_A = 1/(2* np.pi) * np.sqrt(k_A/m)
    
    #set vibrational partition function of the TS to one
    q_TS = 1
    
    # calculate the partition function of A
    q_A = np.exp(- h * nu_A / (2 * R * T)) /  (1 - np.exp(- h * nu_A / ( R * T)) )
    
    k_AB = R * T / h * (q_TS / q_A) * np.exp(- E_b / (R * T))    
    
    return k_AB


def TST_ht_D1(x_A, x_TS, T, m, h, potential): 
    """
    Calculate the rate constant using the high-temperature approximaton of 
    Eyring Transition State Theory (TST) for a unimolecular reaction.

    Parameters:
    - x_A (float): Position of reactant A.
    - x_TS (float): Position of the transition state (TS).
    - T (float): Temperature in Kelvin.
    - m (float): Mass of the reacting molecule.
    - h (float): Discretization interval for numerical Hessian    
    - potential (object): Object representing the potential energy surface, with methods:
        - potential.potential(x): Calculate the potential energy at position x.
        - potential.hessian(x): Calculate the Hessian matrix at position x.
      potential.hessian(x, h) uses the analytical Hessian if implemtend, and the numerical Hessian otherwise 

    Returns:
    - k_AB (float): Rate constant for the reaction from A to B.

    The TST formula used is based on the assumption of a harmonic transition state and the 
    separation of reactant and transition state motions. The vibrational partition function 
    of the transition state (q_TS) is set to one, and the partition function of A (q_A) is 
    approximated by a Taylor expansion. The energy barrier is calculated as (E_b) 
    as the difference in the potential at the transition state and at the reactant state.

    The rate constant (k_AB) is then computed as
    
    k_AB = nu_A * np.exp(- E_b / (R * T))

    The unit of k_AB is 1/ps

    Note: This function assumes a one-dimensional reaction coordinate.
    """
    
    # get natural constants in the right units    
    R = const.R * 0.001
    
    # get the energy barrier
    E_b = potential.potential(x_TS) - potential.potential(x_A)

    #get the force consant in the transition state
    k_A = potential.hessian(x_A, h)[0,0]
    
    # calculate the frequency
    nu_A = 1/(2* np.pi) * np.sqrt(k_A/m)
    
    # rate constant in the high-temperature approximation of Eyring TST
    k_AB = nu_A * np.exp(- E_b / (R * T))    
    
    return k_AB


