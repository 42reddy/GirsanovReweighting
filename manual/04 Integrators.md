
# Overdamped Langevin dynamics
## One-dimensional potentials
The state space for overdamped Langevin dynamics is only the position space $x\in \mathbb{R}$.  
The stochastic differential equation for the time-evolution of the position is

$$
	\dot{x}(t) = + \frac{1}{\xi m}F(x) + \sigma \eta(t)
$$

where $m$ is the mass, $\xi$ is the friction coefficient, $F(x)$ is the force, and $\eta(t)$ is a random process (uncorrelated in time, centred at $\eta=$ with unit variance). The random process is scaled by 
$$
	\sigma = \sqrt{\frac{2RT}{\xi m}} = \sqrt{2D}
$$where $R$ is the ideal gas constant, $T$ is the temperature. The second equality defines the diffusion constant
$$
	D = \frac{RT}{\xi m}
$$
The force is related to the potential energy function by 
$$
	F(x) = -\frac{\mathrm{d}}{\mathrm{d}x}V(x)
$$
#### Euler-Maruyama
The Euler Maruyama algorithm yields a numerical solution for the SDE of overdamped Langevin dynamics: 
$$
	x_{k+1} = x_k + \frac{F(x_k)}{m\xi}\Delta t + \sigma \sqrt{\Delta t} \, \eta_k
$$
where $\Delta t$ is the time step, $x_k$ is the position at time $t=k\Delta t$, $x_{k+1}$ is the position at time $t=(k+1)\Delta t$, and $\eta_k$ is a Gaussian random number (with mean zero and unit variance).

Velocities are not defined for the overdamped Langevin dynamics and are thus not updated.

# Langevin splitting integrators

## One-dimensional potentials
The equations for the individual steps of a Langevin splitting operator, when the state is formulated in terms of coordinates $q_k$ and associated momenta $p_k$, are

$$
\begin{align}
\mathcal{A} \left(\begin{array}{c} q_{k}\\p_{k}\end{array}\right) 
&= \left(\begin{array}{c} q_k+a p_k \\p_k\end{array}\right) \\
%
\mathcal{B}\left(\begin{array}{c} q_{k}\\p_{k}\end{array}\right) 
     &= \left(\begin{array}{c}q_k \\ p_k+b(q_k)\end{array}\right) \\ 
%
\mathcal{O}\left(\begin{array}{c} q_{k}\\ p_{k}\end{array}\right) 
     &= \left(\begin{array}{c} q_k \\ d \, p_k + f\, \eta_k\end{array}\right)
\end{align}
$$

When formulated in terms of positions $x_k$ and associated velocities $v_k$, they are
$$
\begin{align}
\mathcal{A} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
&= \left(\begin{array}{c} x_k+a v_k \\v_k\end{array}\right) \\
%
\mathcal{B}\left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
     &= \left(\begin{array}{c}x_k \\ v_k+b(x_k)\end{array}\right) \\ 
%
\mathcal{O}\left(\begin{array}{c} x_{k}\\ v_{k}\end{array}\right) 
     &= \left(\begin{array}{c} x_k \\ d \, v_k + f_v\, \eta_k\end{array}\right)
\end{align}
$$
with coefficients
$$
\begin{align}
a      &= \Delta t \,\frac{1}{m}\\
b(q_k) &= \Delta t \,F(q_k)\\
b(x_k) &= \Delta t \,F(x_k)\\ \\
d      &= e^{-\xi\Delta t}\\
f      &= \sqrt{RTm\,(1-e^{-2\xi\Delta t})}\\
f_v   &= \frac{1}{m}\sqrt{RTm\,(1-e^{-2\xi\Delta t})} = \sqrt{\frac{RT\,(1-e^{-2\xi\Delta t})}{m}} 
\end{align}
$$
The force is given as 
$$
F(q_k) = -\nabla V(q_k) \, .
$$
$R$ is the ideal gas constant in units $\mathrm{kJ}\,\mathrm{mol}^{-1}\, \mathrm{K}^{-1}$.
If a step is applied with half a time-step, the corresponding operator is denoted with a prime, i.e. $\mathcal{A}'$,$\mathcal{B}'$, $\mathcal{O}'$. 

The system class holds the state of the system and parameters of the dynamics
- $x_k$ position
- $v_k$ velocity
- $m$ mass
- $\Delta t$ time step
- $T$ temperature
- $\xi$ friction coefficient

## Update operators for various splitting algorithms
#### ABO
The ABO algorithm implements the update operator
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}\mathcal{B}\mathcal{A} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### ABOBA
The ABO algorithm implements the update operator
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{B}'\mathcal{O}\mathcal{B}'\mathcal{A}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### AOBOA
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{O}'\mathcal{B}\mathcal{O}'\mathcal{A}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### BAOAB
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{B}'\mathcal{A}'\mathcal{O}\mathcal{A}'\mathcal{B}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$

#### BOAOB
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{B}'\mathcal{O}'\mathcal{A}\mathcal{O}'\mathcal{B}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### OBABO
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}'\mathcal{B}'\mathcal{A}\mathcal{B}'\mathcal{O}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### OABAO
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{O}'\mathcal{A}'\mathcal{B}\mathcal{A}'\mathcal{O}' \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
#### BAOA
$$
\left(\begin{array}{c} x_{k+1}\\v_{k+1}\end{array}\right) =
\mathcal{A}'\mathcal{O}\mathcal{A}'\mathcal{B} \left(\begin{array}{c} x_{k}\\v_{k}\end{array}\right) 
$$
