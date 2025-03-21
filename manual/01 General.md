
# Units

We follow the GROMACS conventions for units:
https://manual.gromacs.org/documentation/2019/reference-manual/definitions.html

For SI units, see
https://en.wikipedia.org/wiki/International_System_of_Units
## Base quantities
| Quantity            | Symbol | Unit                                                                                          | Unit name                          |
| ------------------- | ------ | --------------------------------------------------------------------------------------------- | ---------------------------------- |
| time                | $t$    | $\mbox{ps}= 10^{-12}\, \mbox{s}$                                                              | picosecond                         |
| length              | $x$    | $\mbox{nm} = 10^{-9}\,\mbox{m}$                                                               | nanometer                          |
| mass                | $m$    | $\mbox{u} = 1.66054 \cdot 10^{−27}\,  \mbox{kg} \approx 10^{-3} \frac{\mbox{kg}}{\mbox{mol}}$ | unified atomic mass unit  / Dalton |
| temperature         | $T$    | $\mbox{K}$                                                                                    | Kelvin                             |
| charge              | $q$    | e???                                                                                          | elementary charge ???              |
| collision frequency | $\xi$  | $\mbox{ps}^{-1} = 10^{12}\, \mbox{s}^{-1}$                                                    |                                    |

## Derived quantities
The conversion between GROMACS base units and SI base units, assumes that
$$u \cdot N_A = 0.99999999965(30)\cdot 10^{−3} \frac{\mbox{kg}}{\mbox{mol}} \approx 1$$
where $N_A$ is Avogadro's constant.

| Quantity | Symbol and definition | GROMACS base units | SI base units | Unit |
| ---- | ---- | ---- | ---- | ---- |
| velocity | $v(t) = \frac{d}{dt}x(t)$ | $\frac{\mbox{nm}}{\mbox{ps}}$ | $10^{3} \frac{\mbox{m}}{\mbox{s}}$ |  |
| acceleration | $a(t) = \frac{d}{dt}v(t)$ | $\frac{\mbox{nm}}{\mbox{ps}^2}$ | $10^{15} \frac{\mbox{m}}{\mbox{s}^2}$ |  |
| linear momentum | $p(t) = mv(t)$ | $\mbox{u}\frac{\mbox{nm}}{\mbox{ps}}$ | $\frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}}{\mbox{s}}$ |  |
| force | $F(t) = m a(t)$ | $\mbox{u}\frac{\mbox{nm}}{\mbox{ps}^2}$ | $10^{12} \frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}\cdot\mbox{nm}}$ |
| potential energy | $V(t) = F(t)\Delta x$ | $\mbox{u}\frac{\mbox{nm}^2}{\mbox{ps}^2}$ | $10^{3} \frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}^2}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}}$ |
| kinetic energy | $T(t) = \frac{p^2(t)}{2m} = \frac{1}{2}m v^2(t)$ | $\mbox{u}\frac{\mbox{nm}^2}{\mbox{ps}^2}$ | $10^{3} \frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}^2}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}}$ |

These units are consistent with units for the derivates of the potential energy function:

| Quantity | Symbol and definition | GROMACS base units | SI base units | Unit |
| ---- | ---- | ---- | ---- | ---- |
| potential energy | $V(x)$ | $\mbox{u}\frac{\mbox{nm}^2}{\mbox{ps}^2}$ | $10^{3} \frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}^2}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}}$ |
| force | $F(x) = - \frac{d}{dx}V(x)$ | $\mbox{u}\frac{\mbox{nm}}{\mbox{ps}^2}$ | $10^{12}\frac{\mbox{kg}}{\mbox{mol}}\frac{\mbox{m}}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}\cdot\mbox{nm}}$ |
| hessian | $h(x)= \frac{d^2}{dx^2}V(x)$ | $\mbox{u}\frac{1}{\mbox{ps}^2}$ | $10^{21}\frac{\mbox{kg}}{\mbox{mol}}\frac{1}{\mbox{s}^2}$ | $\frac{\mbox{kJ}}{\mbox{mol}\cdot\mbox{nm}^2}$ |

## Natural constants
| Symbol | Name | Value | Implementation |
| ---- | ---- | ---- | ---- |
| $N_A$ | Avogadro constant | $6.02214076\cdot 10^{23} \, \mathrm{mol}^{-1}$ | scipy.constants.Avogadro |
| $R$ | Ideal gas constant | $8.314462618 \cdot 10^{-3}\, \mathrm{kJ}\, \mathrm{mol}^{-1}\, \mathrm{K}^{-1}$ | scipy.constants.Avogadro * 0.001 |
| $h$ | Planck constant | $0.399031271\,\mathrm{kJ}\,\mathrm{mol}^{−1}\,\mathrm{ps}$ | h = scipy.constants.h *  scipy.constants.h * 0.001 *  1e12 |
|  |  |  |  |

# System

As system is implemented by the system class, which holds information on 

- particle masses
- state of the system, i.e. positions and momenta
- thermodynamic state of the system, i.e. temperature 
- parameters for the interaction with the thermal bath, i.e. collision frequency and simulation time step

## One-dimensional systems

