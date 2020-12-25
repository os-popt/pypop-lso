# Cross-Entropy Method (CEM)

## Reference

```
Duan, Y., Chen, X., Houthooft, R., Schulman, J. and Abbeel, P., 2016, June.
Benchmarking deep reinforcement learning for continuous control.
In International Conference on Machine Learning (pp. 1329-1338).
http://proceedings.mlr.press/v48/duan16.pdf

https://github.com/rll/rllab    (not updated)
https://github.com/rlworkgroup/garage    (actively updated)
https://github.com/rlworkgroup/garage/blob/master/src/garage/np/algos/cem.py    (source code)
```

## Open-Source Implementation

Its source code is openly available at the class [CEM](https://github.com/os-popt/pypop-lso/blob/master/optimizers/cem/cem.py).

In the current implementation, only the diagonal elements ```std``` of covariance matrix of Gaussian sampling distribution (besides ```mean```) are learned and adapted via Maximum Likelihood Estimation (MLE) based on part of fitter samples/individuals (parents). Therefore, the number of learned distribution parameters ```std``` (also called individual step-sizes) equals to the number of decision variables, which can result in a relatively low (even linear) computational complexity for large-scale optimization.

Note that a *delay* update strategy is used for ```std```, in order to encourage more exploration at the early optimization stage.

Furthermore, in the initialization stage only support to set the *global ```std```* (aka global step-size), while the *individual ```std```* (aka individual step-sizes) are updated during optimization.

## Input Arguments

Specific input arguments:

* ```best_frac```: fraction of the best individuals for updating sampling distribution.
  * Float scalar value ranged in (0, 1],
  * If not set, default: 0.05.

* ```init_std```: initial ```std``` (also called global step-size) for sampling distribution.
  * Float scalar value larger than 0,
  * If not set, default: 1.0,
  * Only support isotropic Gaussian sampling distribution during initialization.

* ```extra_std```: std decayed for updating ```std``` of sampling distribution.
  * Float scalar value larger than 0,
  * If not set, default: 1.0,
  * Only support isotropic Gaussian sampling distribution as decayed ```std```.

* ```extra_decay_time```: number of epochs taken to decay std for updating std of sampling distribution.
  * Int scalar value larger than 0,
  * If not set, default: 100.

## Algorithmic Features

The basic algorithmic framework of CEM is very similar to the well-known CMA-ES, though originally they were designed by two different research communities. Probably the biggest difference between CEM and CMA-ES is that the latter uses some more sophisticated heuristics for updating/learning parameters (```mean```+```std``` ) of Gaussian sampling distribution (e.g., evolution path accumulation and covariance matrix decomposition).

## Numerical Experiments

.......
