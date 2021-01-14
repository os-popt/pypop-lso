# Adaptive Sample-Efficient Blackbox Optimization (ASEBO)

## Reference

```
Choromanski, K.M., Pacchiano, A., Parker-Holder, J., Tang, Y. and Sindhwani, V., 2019.
From complexity to simplicity: Adaptive es-active subspaces for blackbox optimization.
In Advances in Neural Information Processing Systems, pp.10299-10309.
https://proceedings.neurips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html

https://github.com/jparkerholder/ASEBO
```

## Open-Source Implementation

Its source code is openly available at the class [ASEBO](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/asebo.py).

Here we include it just *for research purpose* (i.e., for empirical benchmarking).

## Input Arguments

Specific input arguments:

  * ```iota```: number of iterations of full sampling.
    * Integer scalar value larger than 0,
    * If not set, raise a ValueError.

  * ```n_t```: number of samples/individuals (decision vectors / candidate solutions) before ```iota``` iterations/generations.
    * Integer scalar value larger than 0,
    * If not set, default: 100,
    * Before ```iota``` iterations, the actual number of samples is ```2 * n_t``` owing to the used *antithetic sampling* mechanism. After that, it will be adapted automatically during optimization.

  * ```min_n_t```: minimum of samples at each iteration.
    * Integer scalar value larger than 0,
    * If not set, default: 10.

  * ```epsilon```: threshold for PCA.
    * Float scalar value ranged in (0.0, 1.0),
    * If not set, default: 0.995.
  
  * ```sigma```: smoothing parameter for sampling.
    * Float scalar value ranged in (0.0, 1.0],
    * If not set, default: 0.02.

  * ```gamma```: decay rate of covariance matrix adaptation.
    * Float scalar value ranged in (0.0, 1.0),
    * If not set, default: 0.995.

## Algorithmic Features

## Numerical Experiments

.......
