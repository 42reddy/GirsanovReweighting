# Transition state theory

## Eyring TST
The rate constant in Eyring transition state theory is given by
$$k_{AB}^{\mathrm{Eyr}} = \frac{RT}{h} \frac{\widetilde{q}_{TS, \mathrm{vib}}}{q_{A, \mathrm{vib}}} \cdot \frac{q_{TS, \mathrm{rot}}}{q_{A, \mathrm{rot}}} \cdot \frac{q_{TS, \mathrm{trans}}}{q_{A, \mathrm{trans}}} \cdot \exp\left(- \frac{E_b}{RT} \right)$$
where 
- $q_{A, \mathrm{trans}}$ and $q_{TS, \mathrm{trans}}$  are the translational partition function at the minimum $A$ and at the transition state
-  $q_{A, \mathrm{rot}}$  and $q_{TS, \mathrm{rot}}$ are the rotational partition function at the minimum $A$ and at the transition state
-  $q_{A, \mathrm{vib}}$ and $\widetilde{q}_{TS, \mathrm{vib}}$  are the vibrational partition function at the minimum $A$ and at the transition state. The tilde indicates that the reactive vibration has been excluded from the vibrational partition function of the transition state.
- $E_b$ is the energy difference between the minimum and the transition state

In low-dimensional systems, there is now rotational and translational partition function, and the expression for the rate constant reduces to
$$k_{AB}^{\mathrm{Eyr}} = \frac{RT}{h} \frac{\widetilde{q}_{AB^\ddagger, \mathrm{vib}}}{q_{A, \mathrm{vib}}}\cdot \exp\left(- \frac{E_b}{RT} \right)$$
The energy barrier is calculated as
$$ E_b = V(x_{TS}) - V(x_A)\, $$
where $x_{TS}$ is the location of the transition state, and $x_A$ is the location of the reactant state minimum. 

## High-temperature approximation to Eyring TST
The high-temperature approximation to Eyring TST is
$$	k_{AB}^{\mathrm{ht}} 
	= \frac{\prod_{k=1}^{3N-6}  \nu_{A,k}} 	{\prod_{k=1, k\ne r}^{3N-6} \nu_{TS,k}} \cdot \exp\left(-\frac{E_b}{RT}\right)\, . $$
where
- $\nu_{A,k}$ are the vibrational modes in the reactant state $A$
- $\nu_{TS,k}$ are the vibrational modes at the transition state $TS$

The reactive vibrational mode at $\nu_{TS, k=r}$ is excluded from the partition function in Eyring TST and ultimately cancels with a pre-factor. This is why it drops out from the expression for the high-temperature approximation altogether. 
## One-dimensional potentials
### Eyring TST
To evaluate the expression for Eyring TST, we need to calculate
- $q_{TS, \mathrm{vib}}$
- $q_{A, \mathrm{vib}}$

In a one-dimensional potential, there is only one vibrational mode at each stationary point. At the transition state, this vibrational mode is the reactive mode, which in Eyring TST is excluded from partition function of the transition state. Thus, the partition functions at the transition state is set to 1$$q_{TS, \mathrm{vib}} = 1\, .$$The reactant state $A$ is describe by a harmonic approximation, where the one-dimensional potential $V(x)$ is expanded around the minimum at $x=x_A$ in a Taylor series
$$\begin{align}
V(x)\mid_{x=x_A} &= V(x_A) + \frac{d}{d_x}V(x)\mid_{x=x_A} (x-x_A) +\frac{1}{2}\frac{d^2}{dx^2}V(x)\mid_{x=x_A} (x-x_A)^2 + \dots\cr
&= V(x_A) + \frac{1}{2}k (x-x_A)^2 + \dots\cr
\end{align}$$
with 
$$k_A = \frac{d^2}{dx^2}V(x)\mid_{x=x_A}$$
The quantum mechanical eigenstates of this harmonic potential are
$$
	\epsilon_n = h \nu_A  \left(n + \frac{1}{2}\right) = \frac{h}{2\pi}\sqrt{\frac{k_A}{m}}\left(n + \frac{1}{2}\right)
$$
where the frequency is given 
$$
	\nu_A = \frac{1}{2\pi}\sqrt{\frac{k_A}{m}}
$$
The vibrational partition function is
$$
	q_{A, \mathrm{vib}} 
	= \frac{\exp\left(-\frac{h \nu_A}{2RT}\right)}{1 - \exp\left(-\frac{h \nu_A}{RT}\right)}
$$
### High-temperature approximation to Eyring TST
To evaluate the expression for high-temperature approximation of Eyring TST, we need to calculate
- $\prod_{k=1}^{3N-6}  \nu_{A,k}$
- $\prod_{k=1, k\ne r}^{3N-6} \nu_{TS,k}$

Since the vibrational mode is excluded, we set 
$$\prod_{k=1, k\ne r}^{3N-6} \nu_{TS,k}=1 \,.$$
In a one-dimensional potential, there is only one vibrational mode $\nu_A$ in the reactant state and thus
$$\prod_{k=1}^{3N-6}  \nu_{A,k} = \nu_A$$
where $\nu_A$ is defined above.
## Units
- $k$ has units $\frac{\mbox{kJ}}{\mbox{mol}\cdot\mbox{nm}^2}$
- $m$ has units $10^{-3} \frac{\mbox{kg}}{\mbox{mol}}$
- $\nu$ has units $\frac{1}{\mbox{ps}}$
- $h$ has units $\mathrm{kJ}\,\mathrm{mol}^{−1}\,\mathrm{ps}$
- $h\nu$ has units $\mathrm{kJ}\,\mathrm{mol}^{−1}$
- $R$ has units $\mathrm{kJ}\, \mathrm{mol}^{-1}\, \mathrm{K}^{-1}$
- $RT$ has units $\mathrm{kJ}\, \mathrm{mol}^{-1}$
- $\frac{h\nu}{RT}$ is unit-free


# Literature
