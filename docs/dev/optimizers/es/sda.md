# Search Direction Adaptation Evolution Strategy

## Reference

```
He, X., Zhou, Y., Chen, Z., Zhang, J. and Chen, W.N., 2019.
Large-scale evolution strategy based on search direction adaptation.
IEEE Transactions on Cybernetics. (Early Access)
https://ieeexplore.ieee.org/abstract/document/8781905
https://github.com/hxyokokok/SDAES
```

## Open-Source Implementation

## Main Features

When CMA-ES is used for large-scale optimization, the computationally expensive full-parameterized covariance matrix adaptation operator (with *quadratic* time and space complexity) can be replaced by the more efficient **low-rank** model (with *linear* computational complexity). A key assumption of SDA-ES is that the low-rank model can efficiently capture the most promising (local) search directions, even if it cannot capture all pairwise dependencies between decision variables. One of the main innovations of SDA-ES is the construction process of the low-rank model inspired by principal component analysis (PCA), where "pairwise independence" is relaxed to "adjacent independence" for computational efficiency.

It has been empirically demonstrated, especially on quadratic-convex benchmark functions, that SDA-ES is invariant against any affine transformation of the search space (owing to the approximate covariance matrix adaptation via the low-rank model) and the order-preserving transformation of the objective function values (owing to the rank-based selection operator).

## Numerical Experiments

.......

