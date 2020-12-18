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

Its source code is openly available at the class [CEM](https://github.com/os-popt/pypop-lso/blob/master/optimizers/cem/cem.py). Note that in the current implementation, only the diagonal elements of the covariance matrix is learned and adapted via Maximum Likelihood Estimation (MLE) based on part of fitter samples/individuals (parents). Therefore, the number of learned distribution parameters equals to the number of decision variables, which can result in a relatively low (even linear) computational complexity for large-scale optimization.

## Algorithmic Features

The basic algorithmic framework of CEM is very similar to the well-known CMA-ES, though originally they were designed by two different research communities. Probably the biggest difference between CEM and CMA-ES is that the latter uses some more sophisticated heuristics (e.g., evolution path accumulation and covariance matrix decomposition).

## Numerical Experiments

.......
