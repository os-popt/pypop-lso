# Rank-m Evolution Strategy (Rm-ES)

## Reference

```
Li, Z. and Zhang, Q., 2017.
A simple yet efficient evolution strategy for large-scale black-box optimization.
IEEE Transactions on Evolutionary Computation, 22(5), pp.637-646.
https://ieeexplore.ieee.org/abstract/document/8080257
```

## Open-Source Implementation

## Algorithmic Features

Like [R1-ES](https://github.com/os-popt/pypop-lso/blob/master/docs/dev/optimizers/es/r1.md), the main goal of Rm-ES is to use a *sparse plus low-rank* model to reduce the quadratic computational complexity of the *full covariance-matrix* one while still enjoying the well-designed CMA-ES framework. However, different from R1-ES, Rm-ES employs multiple (>=2) promising search directions (rather than a single one) for explorative sampling. Note that any pair of these search directions are expected to be as much independent (i.e., orthogonal) as possible, though the actual implementation does not guarantee such a pairwise independence constraint (note that only an approximate orthogonalization strategy is used in Rm-ES).

## Numerical Experiments

.......
