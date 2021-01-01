# Restart-based Rank-m Evolution Strategy (R-Rm-ES)

## Reference

```
Li, Z. and Zhang, Q., 2017.
A simple yet efficient evolution strategy for large-scale black-box optimization.
IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
https://ieeexplore.ieee.org/abstract/document/8080257
```

## Open-Source Implementation

Its source code is openly available at the class [RestartRm](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/rrm.py).

Note that the **re-start** option is set by default, since generally it provides better performance on complex optimization problems.

## Input Arguments

Specific input arguments:

* ```step_size```: initial global step-size (also called mutation strength) for sampling distribution.
  * Float scalar value larger than 0,
  * If not set, raise a ValueError,
  * Only support isotropic Gaussian sampling distribution during initialization and will be adapted automatically during optimization.

* ```n_evolution_paths```: number of multiple evolution paths (search directions).
  * Integer scalar value larger than 0,
  * If not set, default: 2.

* ```T```: generation gap for multiple evolution paths.
  * Integer scalar value larger than 0,
  * If not set, default: ```problem["ndim_problem"]```.

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

Similar to R1-ES, Rm-ES aims to learn the principal evolution path via the **sparse plus low-rank** model. Different from R1-ES, however, Rm-ES maintains *multiple* search directions (rather than a single one). These multiple search directions are expected to keep orthogonal each other, though in the current implementation such an orthogonalization requirement cannot be guaranteed strictly.

## Numerical Experiments

.......
