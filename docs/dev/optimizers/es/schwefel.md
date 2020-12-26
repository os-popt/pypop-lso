# Schwefel's (1+1)-Evolution Strategy (Schwefel's (1+1)-ES)

## Reference

```
Back, T., Hoffmeister, F. and Schwefel, H.P., 1991, July.
A survey of evolution strategies.
In Proceedings of the Fourth International Conference on Genetic Algorithms (Vol. 2, No. 9).
Morgan Kaufmann Publishers, San Mateo, CA.
```

## Open-Source Implementation

Its source code is openly available at the class [Schwefel](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/schwefel.py).

In the current implementation, only the *global* step-size is used and adapted via Rechenberg's 1/5 success rule during optimization process. The lack of individual step-size adaptation usually limits the performance for complex optimization problems.

*For practical purpose*, we suggest to use more advanced ES versions (e.g., CMA-ES, MA-ES) for possibly high-quality optimization results. Here we include it just *for research purpose* (i.e., for theoretical investigation or for empirical benchmarking).

## Input Arguments

Specific input argument:

* ```step_size```: initial global step-size (also called mutation strength) for sampling distribution.
  * Float scalar value larger than 0,
  * If not set, raise a ValueError,
  * Only support isotropic Gaussian sampling distribution during initialization and will be adapted automatically (via Rechenberg's 1/5 success rule) during optimization.

## Algorithmic Features

Schwefel's (1+1)-ES is one of the earliest evolutionary algorithms for black-box optimization, which motivated lost of improvements, extensions, and variants (such as CMA-ES). For (1+1)-ES, only one parent and one individual are used for each generation, which heavily restrict its power for hard optimization problems.

## Numerical Experiments

.......
