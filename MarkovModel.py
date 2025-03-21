import numpy as np
from scipy import constants

kb = constants.R * 0.001

class MSM():

    def __init__(self, n_states, tau):

        self.n_states = n_states             # number of discrete states
        self.tau = tau                       # path length

    def MSM(self, X, lagtime):

        """
        :param X: simulation path
        :param n_states: number of markov states
        :return: Transition matrix

        """

        bins = np.linspace(min(X), max(X), self.n_states)
        discretized_path = np.digitize(X, bins, right=False) - 1     # assign positions to discrete states

        """
        calculate the count matrix traversing through the discretized simulation path
        """

        count_matrix = np.zeros((self.n_states, self.n_states))

        for i in range(len(discretized_path) - lagtime):
            count_matrix[discretized_path[i], discretized_path[i + lagtime]] += 1

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))              # enforce symmetry

        transition_matrix = count_matrix / np.sum(count_matrix, axis= 1, keepdims=True)    # normalize rows

        return transition_matrix

    def reweighted_MSM(self, X, M, lagtime):

        """
        :param X: simulated path
        :param paths: generated paths from X
        :param M: rewighting factors
        :param lag_time: lag time for the markov model
        :return: reweighted transition matrix
        """

        bins = np.linspace(min(X), max(X), self.n_states + 1)    # bins
        discretized_path = np.zeros(len(X))

        # digitize the path to the bins
        for i in range(self.n_states):
            if i == 0:
                discretized_path[(X >= bins[0]) & (X < bins[1])] = 0
            elif i == self.n_states - 1:
                discretized_path[(X >= bins[self.n_states - 1]) & (X <= bins[self.n_states])] = self.n_states - 1
            else:
                discretized_path[(X >= bins[i]) & (X < bins[i+1])] = i
        discretized_path = discretized_path.astype(int)

        count_matrix = np.zeros((self.n_states, self.n_states))

        len_paths = int(len(X) - lagtime)
        for i in range(len_paths):
            path = discretized_path[i: i + lagtime]

            count_matrix[path[0], path[-1]] += M[i]

        count_matrix = 0.5 * (count_matrix + np.transpose(count_matrix))   # enforce symmetry

        transition_matrix = count_matrix / np.sum(count_matrix, axis=1, keepdims=True)  # normalize rows

        return transition_matrix




