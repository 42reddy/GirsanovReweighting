#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 20:48:01 2024

@author: bettina
"""
#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
import numpy as np
import scipy.constants as const

#---------------------------------------------------------------------
#   O V E R D A M P E D   L A N G E V I N   D Y N A M I C S
#---------------------------------------------------------------------


# Euler-Maruyama integrator
def EM(system, potential, eta_k = None):
    """
    Perform a step according to the Euler-Maruyama integrator for overdamped Langevin dynamics

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'xi_m' (mass * friction coefficient), 
                      'sigma' (standard deviation of the random noise)
                      'dt' (time step) and 'h' (discretization interval for numerical force).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object.
    """    
    # if eta_k is not provided, draw eta_k from Gaussian normal distribution
    if eta_k is None:
        eta_k = np.random.normal()
     
    # update position
    system.x = system.x + (potential.force(system.x, system.h)[0] / system.xi_m ) * system.dt  +  system.sigma * np.sqrt(system.dt) * eta_k
    
    return None  


#---------------------------------------------------------------------
#   B A S I C   L A N G E V I N   I N T E G R A T I O N   S T E P S
#---------------------------------------------------------------------

# A step
def A_step(system, half_step = False):
    """
    Perform the A-step in a Langevin splitting integrator

    Parameters:
        - system (object): An object representing the physical system undergoing Langevin integration.
                  It should have attributes 'x' (position), 'v' (velocity), and 'dt' (time step).
        - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                              a full-step integration. A half-step is often used in the velocity
                              Verlet algorithm for symplectic integration.

    Returns:
        None: The function modifies the 'x' (position) attribute of the provided system object in place.
    """
 
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
        
    system.x = system.x + system.v * dt
    
    return None

# B step
def B_step(system, potential, half_step = False):
    """
    Perform a Langevin integration B-step for a given system.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'v' (velocity), 'm' (mass), and 'x' (position).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                                  a full-step integration. A half-step is often used in the velocity
                                  Verlet algorithm for symplectic integration.

    Returns:
    None: The function modifies the 'v' (velocity) attribute of the provided system object in place based on the
          force calculated by the provided potential object.
    """
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt
    
    system.v = system.v + (1 / system.m) * dt * potential.force_num( system.x, 0.001 )[0]
    
    return None

# O_step
def O_step(system, half_step = False, eta_k = None):
    
    """
    Perform the O-step in a Langevin integrator.

     Parameters:
         - system (object): An object representing the physical system undergoing Langevin integration.
                   It should have attributes 'v' (velocity), 'm' (mass), 'xi' (friction coefficient),
                   'T' (temperature), 'dt' (time step).
         - half_step (bool, optional): If True, perform a half-step integration. Default is False, performing
                   a full-step integration. 
         - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                   in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

     Returns:
     None: The function modifies the 'v' (velocity) attribute of the provided system object in place.
     """

    # get natural constants in the appropriate units    
    R = const.R * 0.001
    
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * system.dt
    else:
        dt = system.dt

    # if eta_k is not provided, draw eta_k from Gaussian normal distribution
    if eta_k is None:
        eta_k = np.random.normal()

    d = np.exp(- system.xi * dt)
    f_v = np.sqrt( R * system.T *  (1 / system.m)  * (1 - np.exp(-2 * system.xi * dt)) )

    system.v = d * system.v +  f_v * eta_k
    return None

#--------------------------------------------------------------
#   L A N G E V I N   S P L I T T I N G   A L G O R I T H M S 
#--------------------------------------------------------------

# ABO integrator
def ABO(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step consisting of A-step, B-step, and O-step.

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system)
    B_step(system, potential)
    O_step(system, eta_k = eta_k)
     
    return None   

# ABOBA integrator
def ABOBA(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the ABOBA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system, half_step= True)
    B_step(system, potential, half_step = True) 
    O_step(system, eta_k = eta_k)    
    B_step(system, potential, half_step = True) 
    A_step(system, half_step = True)
    
    return None   

# ABOBA integrator
def AOBOA(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the AOBOA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    A_step(system, half_step = True)
    O_step(system, half_step = True, eta_k = eta_k)    
    B_step(system, potential) 
    O_step(system, half_step = True, eta_k = eta_k) 
    A_step(system, half_step = True)
    
    return None   
        
# BAOAB integrator
def BAOAB(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the BAOAB algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    B_step(system, potential, half_step = True) 
    A_step(system, half_step = True)
    O_step(system, eta_k = eta_k)    
    A_step(system, half_step = True)
    B_step(system, potential, half_step = True) 
    
    return None   

# BOAOB integrator
def BOAOB(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the BOAOB algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    B_step(system, potential, half_step = True) 
    O_step(system, half_step = True, eta_k = eta_k)   
    A_step(system)   
    O_step(system, half_step = True, eta_k = eta_k)   
    B_step(system, potential, half_step = True)     
    
    return None 

# OBABO integrator
def OBABO(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the OBABO algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    O_step(system, half_step = True, eta_k = eta_k)   
    B_step(system, potential, half_step = True) 
    A_step(system)   
    B_step(system, potential, half_step = True)     
    O_step(system, half_step = True, eta_k = eta_k)   
    
    return None 

# OABAO integrator
def OABAO(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the OABAO algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    O_step(system, half_step = True, eta_k = eta_k)   
    A_step(system, half_step = True)   
    B_step(system, potential, half_step = True) 
    A_step(system, half_step = True)   
    O_step(system, half_step = True, eta_k = eta_k)   
    
    return None 

# BAOA integrator
def BAOA(system, potential, eta_k = None):
    """
    Perform a full Langevin integration step for the BAOA algorithm

    Parameters:
    - system (object): An object representing the physical system undergoing Langevin integration.
                      It should have attributes 'x' (position), 'v' (velocity), 'm' (mass), 'xi'
                      (friction coefficient), 'T' (temperature), and 'dt' (time step).
    - potential (object): An object representing the potential energy landscape of the system.
                         It should have a 'force' method that calculates the force at a given position.
    - eta_k (float or None, optional): If provided, use the given value as the random noise term (eta_k)
                        in the Langevin integrator. If None, draw a new value from a Gaussian normal distribution.

    Returns:
    None: The function modifies the 'x' and 'v' attributes of the provided system object in place.
    """    
    
    B_step(system, potential) 
    A_step(system, half_step = True)   
    O_step(system, eta_k = eta_k)   
    A_step(system, half_step = True)   
    
    return None 


#----------------------------------------------------------------------------
#   B I A S E D   L A N G E V I N   S P L I T T I N G   A L G O R I T H M S 
#----------------------------------------------------------------------------
