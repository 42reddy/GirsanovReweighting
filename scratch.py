from system import system
import matplotlib.pyplot as plt
from potential import D1
from integrator.Biased_integrators import Langevin_integrator
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

doublewell = D1.DoubleWell([1,0,1])
biased = D1.Gaussian_bias([1.5,0,1])


triplewell = D1.TripleWell([4])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
reweighed_bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 2])

y = np.linspace(-2, 2, 200)
potential = bolhuis.potential(y)
potential_reweighted = reweighed_bolhuis.potential(y)
biased_potential = biased.potential(y)

plt.plot(triplewell.potential(y), label='biased')
plt.plot(doublewell.potential(y),label='unbiased')
plt.xlabel('x coordinate')
plt.ylabel('V')
plt.legend(fontsize=8)
plt.title('Simulation vs target potentials')

plt.plot(doublewell.potential(y) - biased.potential(y))


instance = Langevin_integrator(system, 100, 200, doublewell, triplewell)

X, v, eta1, delta_eta1 = instance.generate(int(1e7))
plt.hist(X, bins =1000, density=True)

reweighting_factor = reweighting_factor(system, 100, 200, doublewell, triplewell)
M = reweighting_factor.reweighting_factor(X, eta1, delta_eta1, 200)
plt.plot(M)


markov_state_model = MSM(100, 200)
transition_matrix = markov_state_model.reweighted_MSM(X, M, 200)
transition_matrix = np.nan_to_num(transition_matrix, nan=0)
def equilibrium_dist(transition_matrix):
    A = np.transpose(transition_matrix) - np.eye(100)

    A = np.vstack((A, np.ones(100)))
    b = np.zeros(100)
    b = np.append(b, 1)
    pi = np.linalg.lstsq(A, b, rcond=None)[0]

    return pi


pi = equilibrium_dist(transition_matrix)
x = np.linspace(min(X), max(X),100)
bins = np.linspace(min(X), max(X), 100 + 1, endpoint=True)
x_boltzmann = 0.5 * (bins[1:] + bins[:-1])
boltzmann = np.exp(-triplewell.potential(x_boltzmann)/(kb*T))
x_pi = 0.5 * (bins[1:] + bins[:-1])

plt.plot(x_boltzmann, boltzmann/np.sum(boltzmann), label = 'boltzmann distribution')
plt.plot(x_pi, pi/np.sum(pi), label = 'reweighted eigen vector')
plt.title('underdamped ABOBA , barrier height =4')
plt.xlabel('coordinate along x')
plt.ylabel('probability density')
plt.legend()






eigenvalues, eigenvectors = np.linalg.eig(reweighted_matrix.T)

# Sort eigenvalues and corresponding eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]


# Extract the second and third eigenvectors

second_eigenvector = eigenvectors[:, 1]
plt.plot(-second_eigenvector)


alpha = np.array([2,4,6,8,10,12])
bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, 0])
eigen3 = np.zeros((len(alpha), 9))
eigen2 = np.zeros((len(alpha), 9))
eigen1 = np.zeros((len(alpha), 9))

for i in range(len(alpha)):
    reweighed_bolhuis = D1.Bolhuis([0, 0, 1, 1, 1, alpha[i]])
    instance = underdamped(system, 100, 1000, bolhuis, reweighed_bolhuis)
    X, _, eta, delta_eta = instance.generate(int(5e6))
    lagtimes = np.linspace(10,1500,9)
    eigen3[i,:] = instance.implied_timescales(X, eta, delta_eta, lagtimes)

np.save('eigen61.npy',eigen1)
np.save('eigen62.npy',eigen2)
np.save('eigen63.npy',eigen3)

lagtimes = np.linspace(10,900,9)
# Create the plot

plt.plot(lagtimes, eigen1[0], label ='unbiased')
plt.plot(lagtimes, eigen3[0], label ='reweighted')
plt.plot(lagtimes, eigen2[0], label ='target', linestyle= '--')
plt.xlabel('lagtimes')
plt.ylabel('implied timescales')
plt.title('Implied timescales')
plt.legend()
plt.show()

eigen1 = np.load('eigen21.npy')
eigen2 = np.load('eigen22.npy')
eigen3 = np.load('eigen23.npy')


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(15,10))
ax1.plot(lagtimes, eigen3[0], label='alpha = 2', color = 'g')
ax1.plot(lagtimes, eigen2[0], label='alpha = 2', color ='g', linestyle = '--')
ax1.plot(lagtimes,eigen1[0], label='unbiased',color ='b')

ax1.plot(lagtimes, eigen3[1], label='alpha = 2', color = 'g')
ax1.plot(lagtimes, eigen2[1], label='alpha = 2', color ='g', linestyle = '--')

ax1.plot(lagtimes, eigen3[2], label = 'alpha = 4', color ='b')
ax1.plot(lagtimes, eigen2[2], color = 'b', linestyle = '--' )

ax2.plot(lagtimes, eigen3[3], label = 'alpha = 6', color ='b')
ax2.plot(lagtimes, eigen2[3], color = 'b', linestyle = '--' )

ax2.plot(lagtimes, eigen3[4], label= 'alpha =8', color = 'r')
ax2.plot(lagtimes, eigen2[4], color = 'r' )

ax2.plot(lagtimes, eigen3[5], label = 'alpha =10', color = 'g')
ax2.plot(lagtimes, eigen2[5], color = 'g', linestyle='--')

plt.xlabel('lagtimes')
plt.ylabel('implied timescales')
plt.title('Implied timescales absolute error')
ax1.legend()
ax2.legend()
plt.show()

np.save('eigen11.npy',eigen1)
np.save('eigen21.npy',eigen2)
np.save('eigen31.npy',eigen3)


alpha = np.array([2,4,6,8,10])
plt.plot(bolhuis.potential(y), label='unbiased')
for i in alpha:
    reweighed_bolhuis = D1.Bolhuis([0,0,1,1,1,i])
    plt.plot(reweighed_bolhuis.potential(y), label= 'alpha ='+str(i))
plt.title('target vs simulation potentials')
plt.xlabel(' coordinate')
plt.ylabel('V')
plt.legend()


eigen1 = np.load('eigen11.npy')
eigen2 = np.load('eigen21.npy')
eigen3 = np.load('eigen31.npy')

# Define lagtimes


# Plot the implied timescales for the main plot
plt.figure(figsize=(10, 6))
plt.plot(lagtimes, eigen1[0], label='Unbiased', color='k', linewidth=2)
plt.plot(lagtimes, eigen3[0], label='Reweighted', color='b', linewidth=2)
plt.plot(lagtimes, eigen2[0], label='Target', color='b', linestyle='--', linewidth=2)
plt.xlabel('Lagtimes', fontsize=14)
plt.ylabel('Implied Timescales', fontsize=14)
plt.title('Implied Timescales', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)
plt.show()



x = np.linspace(-2,2,200)
plt.plot((1 / (1 + np.exp(1 * (x + 4)))+ 1 / (1 + np.exp(-1 * (x - 4)))) + (x**2 - 1.2)**2)
plt.plot((x**2 - 1.2)**2)

plt.plot(1 / (1 + np.exp(8 * (x + 1))) * ((x < 0) & (x >= -np.sqrt(2))) )
plt.plot(1 / (1 + np.exp(6 * (x + 1))) * ((x < 0) & (x >= -np.sqrt(2)))  + 1 / (1 + np.exp(-6 * (x - 1))) *((x > 0) & (x <= np.sqrt(2)) ) +
         1 / (1 + np.exp(6 * (-np.sqrt(2) + 1))) * (x < -np.sqrt(2)) + 1 / (1 + np.exp(-6 * (np.sqrt(2) - 1))) * (x > np.sqrt(2)) + (x ** 2 - 2) ** 2)
plt.plot((x ** 2 - 2) ** 2)



