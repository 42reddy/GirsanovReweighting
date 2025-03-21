from system import system
from integrator import D1_stochastic
import numpy as np
from scipy import constants

kb = constants.R * 0.001


class integrator():

    def __init__(self, system, n_states, tau, V_simulation, V_target):
        self.system = system
        self.n_states = n_states             # number of discrete states
        self.tau = tau                       # path length
        self.V_simulation = V_simulation
        self.V_target = V_target

    def gradient(self, x):
        """

        :param x: position
        :return: the gradient of bias potential, bias potential
        """
        return np.array([self.V_simulation.force_ana(x)[0] - self.V_target.force_ana(x)[0],
                         self.V_target.potential(x) - self.V_simulation.potential(x)],
                        dtype=object)




    def ABO(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the ABO algorithm

        Parameters:
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (list or None, optional): A list containing the random noise terms used in the Langevin integrator.
                                          If None, a new value will be drawn from a Gaussian normal distribution.

        Returns:
        delta_eta (float): Random number difference.
        """
        D1_stochastic.A_step(self.system)
        D1_stochastic.B_step(self.system, potential)
        D1_stochastic.O_step(self.system, eta_k=eta_k[0])
        delta_eta = (np.exp(-self.system.xi * self.system.dt) * self.system.dt * (self.gradient(self.system.x)[0])
                     / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt)))))
        return delta_eta

    def AOBOA(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the AOBOA algorithm

        Parameters:
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (list or None, optional): A list containing two random noise terms used in the Langevin integrator.
                                          If None, new values will be drawn from a Gaussian normal distribution.

        Returns:
        delta_eta1, delta_eta2 (float): Random number differences.
        """
        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta1 = (np.exp(-self.system.xi * self.system.dt / 2) +1) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta2 = np.exp(-self.system.xi * self.system.dt / 2) * eta_k[0] + eta_k[1]
        D1_stochastic.A_step(self.system, half_step=True)
        return delta_eta1, delta_eta2

    def BOAOB(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the BOAOB algorithm

        Parameters:
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (list or None, optional): A list containing two random noise terms used in the Langevin integrator.
                                          If None, new values will be drawn from a Gaussian normal distribution.

        Returns:
        delta_eta1, delta_eta2 (float): Random number differences.
        """
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta1 = np.exp(-self.system.xi * self.system.dt / 2) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.A_step(self.system)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta2 = (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)
        return delta_eta1, delta_eta2

    def OBABO(self, potential, eta_k=None):
        """
        Perform a full Langevin integration step for the OBABO algorithm

        Parameters:
        - potential (object): An object representing the potential energy landscape of the system.
                             It should have a 'force' method that calculates the force at a given position.
        - eta_k (list or None, optional): A list containing two random noise terms used in the Langevin integrator.
                                          If None, new values will be drawn from a Gaussian normal distribution.

        Returns:
        delta_eta1, delta_eta2 (float): Random number differences.
        """
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[0])
        delta_eta1 = (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.A_step(self.system)
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, half_step=True, eta_k=eta_k[1])
        delta_eta2 = np.exp(-self.system.xi * self.system.dt / 2) * (self.system.dt / 2) * (self.gradient(self.system.x))[
            0] / (np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-self.system.xi * self.system.dt))))
        return delta_eta1, delta_eta2

    def ABOBA(self, potential, eta_k=None):
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

        D1_stochastic.A_step(self.system, half_step=True)
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.O_step(self.system, eta_k=eta_k[0])
        delta_eta = (np.exp(- self.system.xi * self.system.dt) + 1) * (self.gradient(self.system.x)[0] * self.system.dt / 2) / (
            np.sqrt(kb * self.system.T * self.system.m * (1 - np.exp(-2 * self.system.xi * self.system.dt))))
        D1_stochastic.B_step(self.system, potential, half_step=True)
        D1_stochastic.A_step(self.system, half_step=True)

        return delta_eta


    def generate_ABOBA(self, N):

        """
        simulates the system at simukation potential and langevin integrator
        """
        delta_eta1 = np.zeros(N)   # random number difference
        delta_eta2 = np.zeros(N)
        eta1 = np.zeros(N)       # random number
        eta2 = np.zeros(N)
        X = np.zeros(N)          # positions
        v = np.zeros(N)          # velocities
        X[0] = self.system.x     # initial position
        for i in range(N - 1):
            eta1[i] = np.random.normal(0, 1)  # random number corresponding to random force
            eta2[i] = np.random.normal(0, 1)
            eta = [eta1[i], eta2[i]]
            delta_eta1[i] = self.ABOBA(self.V_simulation, eta_k=eta)
            # update the system
            X[i + 1] = self.system.x
            v[i + 1] = self.system.v

        return X, v, eta1, delta_eta1

    def generate_EM(self, N):

        """
        simulates the system at double well potential and Euler Maruyama scheme
        """
        eta = np.zeros(N)
        delta_eta = np.zeros(N)
        X = np.zeros(N)
        X[0] = self.system.x
        for i in range(N - 1):
            eta[i] = np.random.normal(0, 1)  # random number corresponding to random force
            delta_eta[i] = (np.sqrt(self.system.dt / (2 * kb * self.system.T * self.system.xi * self.system.m)) *
                            self.gradient(self.system.x)[0])
            D1_stochastic.EM(self.system, self.V_simulation, eta_k=eta[i])
            # update the system position
            X[i + 1] = self.system.x

        return X, eta, delta_eta




class Metadynamics():
    def __init__(self, system):
        self.system = system  # system object containing properties like mass, xi_m, dt, and sigma

    def metadynamics(self, N):
        """
        Simulates the system with a double-well potential and a time-dependent bias
        using the Euler-Maruyama scheme.

        Parameters:
        N (int): Number of timesteps for the simulation

        Returns:
        X (np.array): Array of positions over time
        eta (np.array): Array of random forces (stochastic component)
        delta_eta (np.array): Placeholder for future improvements or tracking (currently unused)
        times_added (list): List of positions where bias was added
        bias_potential (np.array): Array to store the incremental bias potential at each timestep
        """

        eta = np.zeros(N)  # Initialize random forces array (Gaussian noise)
        delta_eta = np.zeros(N)  # Initialize array to store the difference in random forces (currently unused)
        X = np.zeros(N)  # Array to store positions of the system
        X[0] = self.system.x  # Initial position of the system (starting from initial condition)
        times_added = []  # List to store the positions where the bias is added
        bias_potential = np.zeros(N)  # Array to store the bias potential at each timestep

        for i in range(N - 1):
            # Random force drawn from a normal distribution
            eta[i] = np.random.normal(0, 1)

            # Double-well potential force (x^2 - 6)^2, which represents the force from the double-well potential
            double_well_force = -6 * (X[i] ** 2 - 1) * X[i]

            # Initialize bias force to zero
            bias_force = 0

            # If times have been added (bias has been applied previously), calculate the bias force
            if len(times_added) > 0:
                # Calculate the difference between the current position X[i] and all previous bias-added positions
                x = X[i] - np.array(times_added)

                # Apply a bias force based on the distance from past bias positions with an exponential factor
                bias_force = 0.001 * np.sum(x * np.exp(-x ** 2 / 0.2) / 0.1)

            # If bias has been added, update the bias potential at the current position
            if len(times_added) > 0:
                # Incremental bias potential as a sum of Gaussian functions centered at previous bias positions
                bias_potential[i] = 1 * np.sum(np.exp(- (X[i] - np.array(times_added)) ** 2 / 0.2))

            # Total force on the particle: sum of the deterministic double-well force and stochastic bias force
            force = double_well_force + bias_force

            # Update the position using the Euler-Maruyama method (Langevin dynamics)
            X[i + 1] = (X[i] +
                        force / self.system.xi_m * self.system.dt +  # Deterministic term: friction/resistance (based on xi_m)
                        self.system.sigma * np.sqrt(self.system.dt) * eta[i])  # Stochastic term: random noise

            # Add a new bias every 20 timesteps (this interval is adjustable)
            if i % 20 == 0:
                times_added.append(X[i])  # Add current position to the list where bias is applied

        # Return final simulation results
        return X, eta, delta_eta, times_added, bias_potential
