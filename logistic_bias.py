from system import system
import matplotlib.pyplot as plt
from potential import D1
from integrator.Biased_integrators import integrator
import numpy as np
from MarkovModel import MSM
from scipy import constants
from Reweighting.Reweighting import reweighting_factor

kb = constants.R * 0.001
m = 1
x = 1.5
v = 0
T = 300
xi = 50
dt = 0.01
h = 0.01

system = system.D1(m,x,v,T,xi,dt,h)

simulation = D1.DoubleWell([1.5, 0, 1])
target = D1.Logistic_bias([1.5, 0, 1])
#target  = D1.Gaussian_bias([1.5, 0, 1])   # similar for the Gaussian bias

# Generates the Underdamped Molecular Dynamics Trajectory
instance = integrator(system, 100, 200, simulation, target)
X, v, eta1, delta_eta1 = instance.generate_ABOBA(int(1e7))

# Calculates the reweighting factor for the shorter individual paths
reweighting_factor = reweighting_factor(system, 100, 200, simulation, target)
M = reweighting_factor.reweighting_factor(X, eta1, delta_eta1, 200)


# Constructs the reweighed Markov State Model
markov_state_model = MSM(100, 200)
transition_matrix = markov_state_model.reweighted_MSM(X, M, 200)
transition_matrix = np.nan_to_num(transition_matrix, nan=0)


# linear regression AX = I, calculates the inverse using LSTSQ
def equilibrium_dist(transition_matrix):
    A = np.transpose(transition_matrix) - np.eye(100)

    A = np.vstack((A, np.ones(100)))
    b = np.zeros(100)
    b = np.append(b, 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    return pi


pi = equilibrium_dist(transition_matrix)     # stationary distribution
bins = np.linspace(min(X), max(X), 100 + 1, endpoint=True)   # Bin edges
x_boltzmann = 0.5 * (bins[1:] + bins[:-1])
boltzmann = np.exp(-target.potential(x_boltzmann)/(kb*T))  # Boltzmann dist at the bin centers
x_pi = 0.5 * (bins[1:] + bins[:-1])

plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution', alpha = 0.7)
plt.plot(x_pi, pi/np.sum(pi), label = 'reweighted eigen vector', alpha = 0.7)
plt.title('overdamped ABOBA, test system')
plt.grid()
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()
plt.show()
