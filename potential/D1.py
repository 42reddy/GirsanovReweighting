#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 07:05:58 2024

@author: bettina
"""

#-----------------------------------------
#   I M P O R T S 
#-----------------------------------------
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import  KMeans


#------------------------------------------------
# abstract class: one-dimensional potentials
#------------------------------------------------
class D1(ABC):
    #---------------------------------------------------------------------
    #   class initialization needs to be implemented in a child class
    #
    #   In the initialization define the parameters of the potential
    #   and the range [x_low, x_high]
    #---------------------------------------------------------------------
    @abstractmethod
    def __init__(self, param):
        pass

    #---------------------------------------------------------------------
    #   analytical functions that need to be implemented in a child class
    #---------------------------------------------------------------------
    # the potential energy function 
    @abstractmethod
    def potential(self, x):
        pass

    # the force, analytical expression
    @abstractmethod
    def force_ana(self, x):
        pass

    # the Hessian matrix, analytical expression
    @abstractmethod
    def hessian_ana(self, x):
        pass

    #-----------------------------------------------------------
    #   numerical methods that are passed to a child class
    #-----------------------------------------------------------
    # negated potential, returns - V(x)
    def negated_potential(self, x):
        """
        Calculate the negated potential energy -V(x) 

        The units of V(x) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float: negated value of the potential energy function at the given position x.
        """
        return -self.potential(x)

        # force, numerical expression via finite difference

    def force_num(self, x, h):
        """
        Calculate the force F(x) numerically via the central finit difference.
        Since the potential is one-idmensional, the force is vector with one element.
        
        The force is given by:
        F(x) = - [ V(x+h/2) - V(x-h/2)] / h
        
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.  
        
        Parameters:
        - x (float): position
 
        Returns:
            numpy array: The value of the force at the given position x , returned as vector with 1 element.  
        """

        F = - (self.potential(x + h / 2) - self.potential(x - h / 2)) / h
        return np.array([F])

    # Hessian matrix, numerical expreesion via second order finite difference
    def hessian_num(self, x, h):
        """
        Calculate the Hessian matrix H(x) numerically via the second-order central finit difference.
        Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
        
        The Hessian is given by:
            H(x) = [V(x+h) - 2 * V(x) + V(x-h)] / h**2
        
        The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
        
        Parameters:
        - x (float): position
        - h (float): spacing of the finit different point along x
        
        Returns:
        numpy array: The 1x1 Hessian matrix at the given position x.
        
        """

        # calculate the Hessian as a float    
        V_x_plus_h = self.potential(x + h)
        V_x = self.potential(x)
        V_x_minus_h = self.potential(x - h)

        H = (V_x_plus_h - 2 * V_x + V_x_minus_h) / h ** 2

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])

        # nearest minimum

    def min(self, x_start):
        """
        Numerically finds the nearest minimum in the vicinity of x_start 
        
        Parameters:
        - x_start (float): start of the minimization
        
        Returns:
        float: position of the minimum
        
        """

        # This is a convenience function.
        # It essentially calls scipy.optimize.minimize.

        # minimize returns a class OptimizeResult
        # the minimum is the class member x
        x_min = minimize(self.potential, x_start, method='BFGS').x

        # returns position of the minimum as float
        return x_min[0]

        # transition state

    def TS(self, x_start, x_end):
        """
        Numerically finds the highest maximum in the interval [x_start, x_end] 
        
        Parameters:
        - x_start (float): position of the reactant minimum
        - x_start (float): position of the product minimum
        
        Returns:
        float: position of the transition state
        
        """

        # find the largest point in [x_start, x_end] on a grid        
        x = np.linspace(x_start, x_end, 1000)
        y = self.potential(x)
        i = np.argmax(y)
        # this is our starting point for the optimization
        TS_start = x[i]

        # minimize returns a class OptimizeResult
        # the transition state is the class member x
        TS = minimize(self.negated_potential, TS_start, method='BFGS').x

        # returns position of the transition state as float
        return TS[0]

        #---------------------------------------------------------------------------------

    #   functions that automatically switch between analytical and numerical function
    #---------------------------------------------------------------------------------    
    # for the force
    def force(self, x, h):
        # try whether the analytical force is implemted
        try:
            F = self.force_ana(x)
        # if force_ana(x) returns a NotImplementedError, use the numerical force instead    
        except NotImplementedError:
            F = self.force_num(x, h)
        return F

    # for the hessian
    def hessian(self, x, h):
        # try whether the analytical hessian is implemted
        try:
            H = self.hessian_ana(x)
        # if hessian_ana(x) returns a NotImplementedError, use the numerical hessian instead    
        except NotImplementedError:
            H = self.hessian_num(x, h)
        return H


#------------------------------------------------
# child class: one-dimensional potentials
#------------------------------------------------

class Constant(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional constant potential based on the given parameter.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: d (float) - constant offset
  
        Raises:
        - ValueError: If param does not have exactly 1 element.
        """

        # Check if param has the correct number of elements
        if len(param) != 1:
            raise ValueError("param must have exactly 1 element.")

        # Assign parameters
        self.d = param[0]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional constant potential.
    
        The potential energy function is given by:
        V(x) = d
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return np.full_like(x, self.d)

    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional constant potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = 0
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = np.full_like(x, 0)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional constant potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 0
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = np.full_like(x, 0)

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Linear(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional linear potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - force constant 
            - param[1]: a (float) - parameter that shifts the extremum left and right

        Raises:
        - ValueError: If param does not have exactly 2 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 2:
            raise ValueError("param must have exactly 2 elements.")

        # Assign parameters
        self.k = param[0]
        self.a = param[1]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional linear potential.
    
        The potential energy function is given by:
        V(x) = k * (x - a) 
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k * (x - self.a)

        # the force, analytical expression

    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional linear potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - k
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = np.full_like(x, -self.k)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional linear potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 0
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = np.full_like(x, 0)

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Quadratic(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional quandratic potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - force constant 
            - param[1]: a (float) - parameter that shifts the extremum left and right

        Raises:
        - ValueError: If param does not have exactly 2 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 2:
            raise ValueError("param must have exactly 2 elements.")

        # Assign parameters
        self.k = param[0]
        self.a = param[1]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional quadratic potential.
    
        The potential energy function is given by:
        V(x) = k * 0.5 * (x-a)**2
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k * 0.5 * (x - self.a) ** 2

        # the force, analytical expression

    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional quadratic potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - k * (x-a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = -self.k * (x - self.a)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional quadratic potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 2k
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = np.full_like(x, self.k)

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class DoubleWell(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional double-well potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - prefactor that scales the potential
            - param[1]: a (float) - parameter that shifts the extremum left and right
            - param[2]: b (float) - parameter controls the separation of the two wells

        Raises:
        - ValueError: If param does not have exactly 3 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 3:
            raise ValueError("param must have exactly 3 elements.")

        # Assign parameters
        self.k = param[0]
        self.a = param[1]
        self.b = param[2]

    # the potential energy function
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional double-well potential.
    
        The potential energy function is given by:
        V(x) = k * ((x-a)**2 - b)**2
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k * ((x - self.a) ** 2 - self.b) ** 2

        # the force, analytical expression

    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional double-well potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - 4 * k * ((x-a)^2 - b) * (x-a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = -4 * self.k * ((x - self.a) ** 2 - self.b) * (x - self.a)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional double-well potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 12 * k * (x-a)^2 - 4 * k * b
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = 12 * self.k * (x - self.a) ** 2 - 4 * self.k * self.b

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class TripleWell(D1):

    def __init__(self, param):
        if len(param) != 1:
            raise ValueError("param must have exactly 1 elements.")

        # Assign parameters
        self.k = param[0]

    def potential(self, x):
        """
               Calculate the potential energy V(x) for the 1-dimensional triple well potential.

        The units of V(x) are kJ/mol, following the convention in GROMACS.

        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k * (x ** 3 - (3 / 2) * x) ** 2 - x ** 3 + x

    def force_ana(self, x):
        """
                Calculate the force F(x) analytically for the 1-dimensional triple well potential.
                Since the potential is one-dimensional, the force is a vector with one element.

                The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.

                Parameters:
                    - x (float): position

                Returns:
                    numpy array: The value of the force at the given position x, returned as vector with 1 element.

                """

        F = -((2 * self.k * (x ** 3 - (3 / 2) * x)) * (3 * x ** 2 - (3 / 2)) - 3 * x ** 2 + 1)

        return np.array([F])

    def hessian_ana(self, x):
        """
                  Calculate the Hessian matrx H(x) analytically for the 1-dimensional triple well potential.
                  Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.

                Parameters:
                    - x (float): position

                  Returns:
                      numpy array: The 1x1 Hessian matrix at the given position x.

                  """

        # calculate the Hessian as a float
        H = 2 * self.k * ((3 * x ** 2 - (3 / 2)) ** 2 + (6 * x) * (x ** 3 - (3 / 2) * x)) - 6 * x

        return np.array([[H]])


class Polynomial(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional polynomial potential (up to order  6) based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: a (float) - parameter that shifts the extremum left and right                
            - param[1]: c1 (float) - parameter for term of order 1
            - param[2]: c2 (float) - parameter for term of order 2
            - param[3]: c3 (float) - parameter for term of order 3
            - param[4]: c4 (float) - parameter for term of order 4
            - param[5]: c5 (float) - parameter for term of order 5
            - param[6]: c6 (float) - parameter for term of order 6            


        Raises:
        - ValueError: If param does not have exactly 7 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 7:
            raise ValueError("param must have exactly 7 elements.")

        # Assign parameters
        self.a = param[0]
        self.c1 = param[1]
        self.c2 = param[2]
        self.c3 = param[3]
        self.c4 = param[4]
        self.c5 = param[5]
        self.c6 = param[6]

    # the potential energy function
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional polynomial potential.
    
        The potential energy function is given by:
        V(x) = c1*(x-a) + c2*(x-a)**2 + c3*(x-a)**3 + c4*(x-a)**4 + c5*(x-a)**5 + c6*(x-a)**6 
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.c1 * (x - self.a) + self.c2 * (x - self.a) ** 2 + self.c3 * (x - self.a) ** 3 + self.c4 * (
                    x - self.a) ** 4 + self.c5 * (x - self.a) ** 5 + self.c6 * (x - self.a) ** 6

    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional polynomial potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - c1 - 2*c2*(x-a) - 3*c3*(x-a)**2 - 4*c4*(x-a)**3 - 5*c5*(x-a)**4 - 6*c6*(x-a)**5 
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = - self.c1 - 2 * self.c2 * (x - self.a) - 3 * self.c3 * (x - self.a) ** 2 - 4 * self.c4 * (
                    x - self.a) ** 3 - 5 * self.c5 * (x - self.a) ** 4 - 6 * self.c6 * (x - self.a) ** 5
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional polynomial potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 2*c2 + 6*c3*(x-a)+ 12*c4*(x-a)**2 + 20*c5*(x-a)**3 + 30*c6*(x-a)**4 
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = 2 * self.c2 + 6 * self.c3 * (x - self.a) + 12 * self.c4 * (x - self.a) ** 2 + 20 * self.c5 * (
                    x - self.a) ** 3 + 30 * self.c6 * (x - self.a) ** 4

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Bolhuis(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional Bolhuis potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: a (float) - parameter controlling the center of the quadratic term.
            - param[1]: b (float) - parameter controlling the width of the quadratic term.
            - param[2]: c (float) - parameter controlling the width of perturbation.
            - param[3]: k1 (float) - force constant of the double well. Default is 1.d
            - param[4]: k2 (float) - force constant of the linear term. Default is 0.
            - param[5]: alpha (float) - strength of the perturbation.


        Raises:
        - ValueError: If param does not have exactly 6 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 6:
            raise ValueError("param must have exactly 6 elements.")

        # Assign parameters
        self.a = param[0]
        self.b = param[1]
        self.c = param[2]
        self.k1 = param[3]
        self.k2 = param[4]
        self.alpha = param[5]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional Bolhuis potential.
    
        The potential energy function is given by:
        V(x) = k1 * ((x - a)**2 - b)**2 + k2 * x + alpha * np.exp(-c * (x - a)**2)
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k1 * ((x - self.a) ** 2 - self.b) ** 2 + self.k2 * x + self.alpha * np.exp(
            -self.c * (x - self.a) ** 2)

    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional Bolhuis potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - 2 * k1 * ((x - a)**2 - b) * 2 * (x-a) - k2 + alpha * np.exp(-c * (x - a)**2) * c * 2 * (x - a)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = - 2 * self.k1 * ((x - self.a) ** 2 - self.b) * 2 * (x - self.a) - self.k2 + self.alpha * np.exp(
            -self.c * (x - self.a) ** 2) * self.c * 2 * (x - self.a)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional Bolhuis potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = 12 * k1 (x - a)**2   +   4 * k1 * b   +   2 * alpha * c * [ 4 * c * (x-a)**2 - (x-a)] * exp (-c *(x-2)**2 )
    
          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = 12 * self.k1 * (x - self.a) ** 2 - 4 * self.k1 * self.b + 2 * self.alpha * self.c * (
                    2 * self.c * (x - self.a) ** 2 - 1) * np.exp(-self.c * (x - self.a) ** 2)

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Prinz(D1):
    # intiialize class
    def __init__(self):
        """
        Initialize the class for the 1-dimensional Prinz potential. 
        All parameters are hard-coded.
        """

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional constant potential.
    
        The potential energy function is given by:
        V(x) = 4* ( x^8 + 0.8 * e^(-80x^2) + 0.2 * e^(-80(x-0.5)^2) + 0.5 * e^(-40(x+0.5)^2) )
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return 4 * (x ** 8 + 0.8 * np.exp(-80 * x ** 2) + 0.2 * np.exp(-80 * (x - 0.5) ** 2) + 0.5 * np.exp(
            - 40 * (x + 0.5) ** 2))

    # the force, analytical expression 
    def force_ana(self, x):
        """
        The class method force_ana(x) is not implemented in the class for the Prinz potential. 
        Use force_num(x,h) instead.
        """

        raise NotImplementedError("potential.D1(Prinz) does not implement force_ana(self, x)")

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
        The class method hessian_ana(x) is not implemented in the class for the Prinz potential. 
        Use hessian_num(x,h) instead.
        """

        raise NotImplementedError("potential.D1(Prinz) does not implement hessian_ana(self, x)")


class Logistic(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional Logistic potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - prefactor that scales the potential
            - param[1]: b (float) - logistic growth rate or slope of the curve
            - param[2]: a (float) - parameter that shifts the saddle point left and right


        Raises:
        - ValueError: If param does not have exactly 3 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 3:
            raise ValueError("param must have exactly 3 elements.")

        # Assign parameters
        self.k = param[0]
        self.b = param[1]
        self.a = param[2]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional Logistic potential.
    
        The potential energy function is given by:
        V(x) = k * 1 /(1 +  exp(-b * (x - a))
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k * 1 / (1 + np.exp(- (self.b * (x - self.a))))

        # the force, analytical expression

    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional Logistic potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = - k * (b * exp(-b * (x-a))) / (exp(-b * (x - a)) + 1)^2
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = - self.k * (self.b * np.exp(-self.b * (x - self.a))) / (np.exp(- (self.b * (x - self.a))) + 1) ** 2
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional Logistic potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = - kb^2 * ((2 * exp(-2b (x - a))) / (1 + exp(-b * (x - a)))^3 - (exp(-b * (x - a))) / ((1 + exp(-b * (x - a))))^2) )

          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = self.k * self.b ** 2 * (
                    (2 * np.exp(-2 * self.b * (x - self.a))) / (1 + np.exp(-self.b * (x - self.a))) ** 3 - (
                np.exp(-self.b * (x - self.a))) / ((1 + np.exp(-self.b * (x - self.a)))) ** 2)

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Gaussian(D1):
    # intiialize class
    def __init__(self, param):
        """
        Initialize the class for the 1-dimensional Logistic potential based on the given parameters.

        Parameters:
            - param (list): a list of parameters representing:
            - param[0]: k (float) - prefactor that scales the potential
            - param[1]: mu (float) - parameter that shifts the extremum left and right (mean)
            - param[2]: sigma (float) - parameter that determines the broadening of the Gaussian (variance)


        Raises:
        - ValueError: If param does not have exactly 3 elements.
        """

        # Check if param has the correct number of elements
        if len(param) != 3:
            raise ValueError("param must have exactly 3 elements.")

        # Assign parameters
        self.k = param[0]
        self.mu = param[1]
        self.sigma = param[2]

    # the potential energy function 
    def potential(self, x):
        """
        Calculate the potential energy V(x) for the 1-dimensional Logistic potential.
    
        The potential energy function is given by:
        V(x) = k / (sqrt(2 sigma^2 pi)) * exp(-(x - mu)^2 / (2 sigma^2))
    
        The units of V(x) are kJ/mol, following the convention in GROMACS.
    
        Parameters:
            - x (float): position

        Returns:
            float: The value of the potential energy function at the given position x.
        """

        return self.k / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2))

    # the force, analytical expression 
    def force_ana(self, x):
        """
        Calculate the force F(x) analytically for the 1-dimensional Logistic potential.
        Since the potential is one-dimensional, the force is a vector with one element.
    
        The force is given by:
        F(x) = - dV(x) / dx 
             = k / (sqrt(2 pi) sigma^3) * exp(-(x - mu)^2 / (2 sigma ^2)) * (x - mu)
    
        The units of F(x) are kJ/(mol * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position
    
        Returns:
            numpy array: The value of the force at the given position x, returned as vector with 1 element.
    
        """

        F = self.k / (self.sigma ** 3 * np.sqrt(2 * np.pi)) * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) * (
                    x - self.mu)
        return np.array([F])

    # the Hessian matrix, analytical expression
    def hessian_ana(self, x):
        """
          Calculate the Hessian matrx H(x) analytically for the 1-dimensional Logistic potential.
          Since the potential is one dimensional, the Hessian matrix has dimensions 1x1.
    
          The Hessian is given by:
          H(x) = d^2 V(x) / dx^2 
               = k / (sqrt(2 pi) sigma^3) * ( exp(-(x - mu)^2 / (2 sigma ^2)) - (exp(-(x - mu)^2 / (2 sigma ^2)) * (x - mu)^2 / (sigma^2))

          The units of H(x) are kJ/(mol * nm * nm), following the convention in GROMACS.
    
        Parameters:
            - x (float): position

          Returns:
              numpy array: The 1x1 Hessian matrix at the given position x.
    
          """

        # calculate the Hessian as a float
        H = - self.k / (self.sigma ** 3 * np.sqrt(2 * np.pi)) * (np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) - (
                    (x - self.mu) ** 2 * np.exp(-(x - self.mu) ** 2 / (2 * self.sigma ** 2)) / self.sigma ** 2))

        # cast Hessian as a 1x1 numpy array and return
        return np.array([[H]])


class Gaussian_bias(D1):

    def __init__(self, param):
        self.h = param[0]  # Scaling factor for the potential
        self.well_seperation = 2 * param[2]  # Double the value of the well separation
        self.sigma = self.well_seperation / 6  # Calculate sigma based on well separation (standard deviation for Gaussian)

    def potential(self, x):
        """

        Parameters:
        x (float): Position at which to calculate the potential

        Returns:
        float: The total potential at the position `x`
        """

        double_well = self.h * (x ** 2 - 1) ** 2

        # Gaussian bias potential: sum of two Gaussian functions centered at x = ±1
        bias = 0.4 * self.h * (
                    np.exp(-(x - 1) ** 2 / (2 * self.sigma ** 2)) + np.exp(-(x + 1) ** 2 / (2 * self.sigma ** 2)))

        return double_well + bias

    def force_ana(self, x):
        """

        Parameters:
        x (float): Position at which to calculate the force

        Returns:
        np.array: The force at the given position `x`
        """

        double_well_force = -4 * self.h * (x ** 2 - 1) * x

        # Gaussian bias force: derivative of the Gaussian terms with respect to x
        bias_force = -0.4 * self.h * (
                -((x + 1) / (self.sigma ** 2)) * np.exp(-((x + 1) ** 2) / (2 * self.sigma ** 2))
                - ((x - 1) / (self.sigma ** 2)) * np.exp(-((x - 1) ** 2) / (2 * self.sigma ** 2))
        )

        force = double_well_force + bias_force

        return np.array([force])

    def hessian_ana(self, x):
        """
        Parameters:
        x (float): Position at which to calculate the Hessian

        Returns:
        np.array: 1x1 Hessian matrix (since we are in 1D, the Hessian is a scalar)
        """

        double_well_hessian = -12 * self.h * x ** 2 + 4 * self.h

        # Second derivative of the Gaussian bias term:
        term1_second_derivative = -2 * (x + 1) * np.exp(-((x + 1) ** 2) / (2 * self.sigma ** 2)) / self.sigma ** 2
        term2_second_derivative = -2 * (x - 1) * np.exp(-((x - 1) ** 2) / (2 * self.sigma ** 2)) / self.sigma ** 2

        # Bias Hessian term: the second derivative of the Gaussian bias potential with respect to x
        bias_hessian = -0.4 * self.h * (term1_second_derivative + term2_second_derivative)

        hessian = double_well_hessian + bias_hessian

        return np.array([hessian])


class Logistic_bias(D1):

    def __init__(self, param):

        self.h = param[0]  # Scaling factor for the potential
        self.well_seperation = 2 * param[2]  # Double the value of the well separation
        self.sigma = self.well_seperation / 6  # Calculate sigma based on well separation (standard deviation for Gaussian)

    def potential(self, x):
        """
        Parameters:
        x (float): Position at which to calculate the potential

        Returns:
        float: The total potential at the position `x`
        """


        double_well = self.h * (x**2 - 1) ** 2

        # Logistic bias potential: combination of two logistic functions centered at x = ±1
        bias = self.h * ((1 / (1 + np.exp(-(x - 1)))) + (1 / (1 + np.exp(x + 1)))) * x ** 2


        return double_well + bias

    def force_ana(self, x):
        """
        Parameters:
        x (float): Position at which to calculate the force

        Returns:
        np.array: The force at the given position `x`

        """


        double_well_force = -4 * self.h * (x ** 2 - 1) * x

        # Derivative of the logistic terms with respect to x (Bias Force)
        term1_derivative = np.exp(-(x - 1)) / (1 + np.exp(-(x - 1))) ** 2
        term2_derivative = -np.exp(x + 1) / (1 + np.exp(x + 1)) ** 2

        # Bias force: derivative of the logistic bias potential with respect to x
        bias_force = -self.h * ((term1_derivative + term2_derivative) * x ** 2 +
                                2 * ((1 / (1 + np.exp(-(x - 1)))) + (1 / (1 + np.exp(x + 1)))) * x)

        force = double_well_force + bias_force

        return np.array([force])


    def hessian_ana(self, x):
        """
        Parameters:
        x (float): Position at which to calculate the Hessian

        Returns:
        np.array: 1x1 Hessian matrix (since we are in 1D, the Hessian is a scalar)
        """


        double_well_hessian = -12 * self.h * x ** 2 + 4 * self.h


        term1_second_derivative = 2 * np.exp(-(x - 1)) * (np.exp(-(x - 1)) - 1) / (1 + np.exp(-(x - 1))) ** 3
        term2_second_derivative = -2 * np.exp(x + 1) * (np.exp(x + 1) - 1) / (1 + np.exp(x + 1)) ** 3

        # Bias Hessian term: the second derivative of the logistic bias potential with respect to x
        bias_hessian = -self.h * ((term1_second_derivative + term2_second_derivative) * x ** 2 +
                                  4 * x + 4 * (x * (term1_second_derivative + term2_second_derivative)))

        hessian = double_well_hessian + bias_hessian

        return np.array([hessian])




class Metadynamics_bias(D1):
    def __init__(self, times_added, barrier_height, poly_degree=10):
        self.times_added = np.array(times_added)
        self.poly_degree = poly_degree
        self.barrier_height = barrier_height

        self.potential_poly = self._fit_polynomial_potential()

    def meta_potential(self, x):

        """Computes the bias potential from metadynamics using Gaussians."""

        return (self.barrier_height / len(self.times_added)) * np.sum(np.exp(- (x - self.times_added) ** 2 / 0.02)) + 1.5 * (x**2 - 1)**2 - 2


    def _fit_polynomial_potential(self):

        """Fits a polynomial dynamically based on the added Gaussians."""

        x_vals = np.linspace(min(self.times_added), max(self.times_added), 500)
        y_vals = np.array([self.meta_potential(x) for x in x_vals])

        return np.poly1d(np.polyfit(x_vals, y_vals, self.poly_degree))

    def potential(self, x):

        """Evaluates the fitted polynomial potential at x."""

        return self.potential_poly(x)

    def force_ana(self, x):

        """Computes the analytical force (negative gradient of potential) at x."""

        force_poly = np.polyder(self.potential_poly)
        return np.array([-force_poly(x)])


    def hessian_ana(self, x):

        """Computes the analytical Hessian (second derivative of potential)."""

        hessian_poly = np.polyder(self.potential_poly, 2)
        return np.array([[hessian_poly(x)]])






