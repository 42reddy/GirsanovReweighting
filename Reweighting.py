import numpy as np
from scipy import constants
kb = constants.R * 0.001


class reweighting_factor():

    def __init__(self, system, n_states, tau, V_simulation, V_target):

        self.system = system
        self.n_states = n_states             # number of discrete states
        self.tau = tau                       # path length
        self.V_simulation = V_simulation    #
        self.V_target = V_target


    def reweighting_factor(self, X, eta, delta_eta, lagtime, eta1=None, delta_eta1=None):

        """
        :param delta_eta1: second random number at target potential
        :param eta1: second random number at simulation
        :param eta: random number at simulation potential
        :param delta_eta: random number at target potential
        :param X: simulation path
        :return: reweighting factors for each path
        """

        len_paths = int((len(X) - lagtime))  # length of the paths generated from sliding window method

        """ calculate the reweighting factor for each observed path """
        M = np.zeros(len_paths)
        for i in range(len_paths):
            """calculate eta and delta_eta for each path"""
            eta_ = eta[i : i + lagtime ]
            delta_eta_ = delta_eta[i : i + lagtime]

            """calculate the reweighting factor"""

            if delta_eta1 is not None:    # switch between schemes with single and two random numbers
                eta_1 = eta1[i: i + lagtime]
                delta_eta_1 = delta_eta1[i: i + lagtime]

                M[i] = np.exp((self.V_simulation.potential(X[i]) - self.V_target.potential(X[i])) / (kb * self.system.T)) * (
                           np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2))) * (
                           np.exp(-np.sum(eta_1 * delta_eta_1)) * np.exp(-0.5 * np.sum(delta_eta_1 ** 2)))
            else:
                M[i] = np.exp((self.V_simulation.potential(X[i]) - self.V_target.potential(
                    X[i])) / (kb * self.system.T)) * (np.exp(-np.sum(eta_ * delta_eta_)) * np.exp(-0.5 * np.sum(delta_eta_ ** 2)))

        return M