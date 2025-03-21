# 1-dimensional potentials

## Class D1

Class D1 is the parent class for all one-dimensional potentials. It provides the following a **abstract mehods** which need to be implemented for each specific potential $V(x)$.

- **\_\_init\_\_**:  set the parameters of the potential
- **potential**: expects $x$, returns $V(x)$, should be implemented as an analytical expression
- **force_ana:** expects x, returns $-\mathrm{d}V(x)/\mathrm{d}x$, should be implemented as an analytical expression
- **hessian_ana**: expects x, returns  $\mathrm{d}^2V(x)/\mathrm{d}x^2$, should be implemented as an analytical expression

The also class provides methods that are inherited by the child classes

- **negated_potential**: expects $x$, calls method potential, returns $-V(x)$
- **force_num**: expects $x$ and $h$, calls method potential, returns $-\mathrm{d}V(x)/\mathrm{d}x$ calculated via finite difference
- **hessian_num**: expects $x$ and $h$, calls method potential, returns  $\mathrm{d}^2V(x)/\mathrm{d}x^2$ calculated via finite difference
- **min:** expects $x_{\mathrm{start}}$, returns location of the nearest minimum calculated via scipy.optimize
- **TS:** expects $x_{\mathrm{start}}$ and $x_{\mathrm{end}}$, returns location of the highest energy point in the intervall $[x_{\mathrm{start}}, x_{\mathrm{end}}]$$ (i.e. the transition state), calculated via scipy.optimize. Returns error, if the there is no energy maximum in the interval.

The class provides function that automatically switch between analytical and numerical implementation
- **force:** expects $x$ and $h$, calls force_ana(x) if this is implemented and, force_num(x,h) otherwise. 
- **hessian:** expects $x$ and $h$, calls hessian_ana(x) if this is implemented and, hessian_num(x,h) otherwise.  

In this way, MD integrators and rate models etc. can be implemented using force(x,h) and hessian(x,h). The program than also runs on potentials for which the analytical force and the analytical Hessian have not been implemented.

## Constant potential
**Parameters:** $d$
**Potential:**
$$V(x) = d$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = 0$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 0$$
## Linear potential

**Parameters:** $k, a$
**Potential:**
$$V(x) = k(x-a)$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = k$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 0$$
## Quadratic potential

**Parameters:** $k, a$
**Potential:**
$$V(x) = \frac{k}{2} (x-a)^2 $$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = k (x-a)$$

**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = k$$
## Morse potential

**Parameters:** $D_e, a, x_e$
**Potential:**
$$V(x) = D_e \left(1- e^{-a(x-x_e) } \right)^2$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = -2a D_e \left(e^{-a(x-x_e)}- e^{-2a(x-x_e)} \right)$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 2a^2 D_e \left(2e^{-2a(x-x_e)}-e^{-a(x-x_e)} \right)$$
## Lennard-Jones potential

**Parameters:** $\epsilon, \sigma$
**Potential:**
$$V(x) =  4\epsilon \left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6 \right]$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = 24\epsilon \left[2\left(\frac{\sigma}{r}\right)^{12}\frac{1}{r}-\left(\frac{\sigma}{r}\right)^6\frac{1}{r} \right]$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 24\epsilon \left[26\left(\frac{\sigma}{r}\right)^{12}\frac{1}{r^2}-7\left(\frac{\sigma}{r}\right)^6\frac{1}{r^2} \right]$$

## Gaussian potential

**Parameters:** $k, \mu, \sigma$
**Potential:**
$$V(x) = \frac{k}{\sqrt{2\sigma^2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) $$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = + \frac{k}{\sqrt{2\pi}\sigma^3} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\cdot(x-\mu)$$

**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = - \frac{k}{\sqrt{2\pi}\sigma^3} \left(\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)-\frac{\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\cdot (x-\mu)^2}{\sigma^2} \right)$$


## Logistic potential

**Parameters:** $k, a, b$
**Potential:**
$$V(x) = k\cdot \frac{1}{1+e^{-b(x-a)}}$$

**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = -k\cdot\frac{be^{-b(x-a)}}{\left(e^{-b(x-a)}+1\right)^2}$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = k\cdot\left(\frac{2b^2e^{-2b(x-a)}}{(1+e^{-b(x-a)})^3}- \frac{b^2e^{-b(x-a)}}{(1+e^{-b(x-a)})^2} \right)$$



## Double well

**Parameters:** $k, a, b$
**Potential:**
$$V(x) = k \cdot ((x - a)^2 - b)^2$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = - 4 k\cdot ((x-a)^2-b)\cdot(x-a)$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 12 k \cdot (x-a)^2- 4 k \cdot b $$



## Polynomial potential

**Parameters:** $a, c_1, c_2, c_3, c_4, c_5, c_6$
**Potential:**
$$V(x) = c_6(x-a)^6 + c_5(x-a)^5 + c_4(x-a)^4 + c_3(x-a)^3 + c_2(x-a)^2+ c_1(x-a)$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = - 6c_6(x-a)^5 - 5c_5(x-a)^4 - 4c_4(x-a)^3 - 2c_3(x-a)^2 - 2c_2(x-a) - c_1$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 30c_6(x-a)^4 + 20c_5(x-a)^3 + 12c_4(x-a)^2 + 4c_3(x-a) + 2c_2$$

## Bolhuis potential

**Parameters:** $k_1, k_2, a, b$
**Potential:**
$$V(x) = k_1 \cdot ((x - a)^2 - b)^2 + k_2 \cdot x + \alpha e^{-c(x-a)^2}$$
**Force:**
$$F(x) = - \frac{\mathrm{d}}{\mathrm{d}x} V(x) = - 4 k_1\cdot ((x-a)^2-b)\cdot(x-a)  - k_2 + \alpha e^{-c(x-a)^2} \cdot2c(x-a)$$
**Hessian:**
$$H_{11}(x) = \frac{\mathrm{d}^2}{\mathrm{d}x^2}V(x) = 12 k_1 \cdot (x-a)^2- 4 k_1 \cdot b 
			+ 2 \alpha c \cdot [4c (x-a)^2 - (x-a)] \cdot e^{-c (x-2)^2 }
$$



## Prinz potential

From JH Prinz et.al. The Journal of chemical physics, 134(17):174105, 2011. [https://doi.org/10.1063/1.3565032](https://doi.org/10.1063/1.3565032)
https://deeptime-ml.github.io/trunk/api/generated/deeptime.data.prinz_potential.html#deeptime.data.prinz_potential

**Parameters:** None. Parameters are hard-coded.
**Potential:**
$$V(x) = 4\left(x^8 + 0.8e^{-80x^2} + 0.2e^{-80(x-0.5)^2} + 0.5 e^{-40(x+0.5)^2}\right)$$
**Force:**
Let's not calculate this, and use the numerical force instead. 

**Hessian:**
Let's not calculate this, and use the numerical Hessian instead. 


# 2-dimensional potentials
# 3-dimensional potentials
# n-dimensional potentials

