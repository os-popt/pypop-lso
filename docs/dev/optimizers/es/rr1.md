# Restart-based Rank-One Evolution Strategy (R-R1-ES)

## Reference

```
Li, Z. and Zhang, Q., 2017.
A simple yet efficient evolution strategy for large-scale black-box optimization.
IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
https://ieeexplore.ieee.org/abstract/document/8080257
```

## Open-Source Implementation

Its source code is openly available at the class [RestartRankOne](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/rr1.py).

Note that the **re-start** option is set by default, since generally it provides better performance on complex optimization problems.

## Input Arguments

Specific input arguments:

* ```step_size```: initial global step-size (also called mutation strength) for sampling distribution.
  * Float scalar value larger than 0,
  * If not set, raise a ValueError,
  * Only support isotropic Gaussian sampling distribution during initialization and will be adapted automatically during optimization.

* ```c_cov```: learning (changing) rate of covariance matrix adaptation.
  * Float scalar value larger than 0,
  * If not set, default: ```1 / (3 * np.sqrt(problem["ndim_problem"]) + 5)```.

* ```c```: learning (changing) rate of principal search direction (also called evolution path).
  * Float scalar value larger than 0,
  * If not set, default: ```2 / (problem["ndim_problem"] + 7)```.

* ```c_s```: learning (changing) rate of cumulative rank rate.
  * Float scalar value larger than 0,
  * If not set, default: 0.3.

* ```q_star```: target ratio for mutation strength adaptation.
  * Float scalar value larger than 0,
  * If not set, default: 0.3.

* ```d_sigma```: damping factor for mutation strength adaptation.
  * Float scalar value larger than 0,
  * If not set, default: 1.

## Algorithmic Features

Although its basic algorithmic framework is still based on the well-designed CMA-ES, R-R1-ES has some obvious advantages over CMA-ES (e.g., computationally much more efficient sampling and covariance matrix approximating/updating) especially for large-scale, black-box optimization. The core idea of R-R1-ES is to use a *sparse plus low-rank* sampling model with linear computational complexity (rather than the *full covariance-matrix* one with quadratic computational complexity) to capture the promising/principal search direction.

## Numerical Experiments

.......
