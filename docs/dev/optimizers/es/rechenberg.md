# Rechenberg's (1+1)-Evolution Strategy (Rechenberg's (1+1)-ES)

## Reference

```
Back, T., Hoffmeister, F. and Schwefel, H.P., 1991, July.
A survey of evolution strategies.
In Proceedings of the Fourth International Conference on Genetic Algorithms (Vol. 2, No. 9).
Morgan Kaufmann Publishers, San Mateo, CA.
```

## Open-Source Implementation

Its source code is openly available at the class [Rechenberg](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/rechenberg.py).

In the current implementation, only the *global* step-size is used and keeps **fixed** during optimization process. The lack of step-size adaptation usually means that such a hyperparameter needs to be carefully fine-tuned (e.g., via grid search) for satisfactory performance even in simple optimization problems.

*For practical purpose*, we suggest to use more advanced ES versions (e.g., CMA-ES, MA-ES) for possibly high-quality optimization results. Here we include it just *for research purpose* (i.e., for theoretical investigation or for empirical benchmarking).

## Input Arguments

Specific input argument:

* ```step_size```: initial global step-size (also called mutation strength) for sampling distribution.
  * Float scalar value larger than 0,
  * If not set, raise a ValueError,
  * Only support isotropic Gaussian sampling distribution during initialization and always keep fixed during optimization.

## Algorithmic Features

Rechenberg's (1+1)-ES is one of the earliest evolutionary algorithms for black-box optimization, which motivated lost of improvements, extensions, and variants (such as CMA-ES). For (1+1)-ES, only one parent and one individual are used for each generation.

## Numerical Experiments

.......
