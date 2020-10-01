# Ostermeier's (1,lambda)-Evolution Strategy

## Reference

```
Ostermeier, A., Gawelczyk, A. and Hansen, N., 1994.
A derandomized approach to self-adaptation of evolution strategies.
Evolutionary Computation, 2(4), pp.369-380.
https://www.mitpressjournals.org/doi/10.1162/evco.1994.2.4.369
```

## Open-Source Implementation

In the library ```pypop-lso```, the source code of **Ostermeier's (1,lambda)-Evolution Strategy** is available at the ES class [Ostermeier](https://github.com/os-popt/pypop-lso/blob/master/optimizers/es/ostermeier.py).

For its current implementation, unstable performance was observed on even benchmark functions chosen by the original paper (see the below experiments for details). Such an instability is caused by the stochastic fluctuations of the individual step-sizes over the adaptation process. As a result, it's not suggested to use it *for product purpose*. Instead, other more sophisticated ES versions (e.g., CMA-ES) should be considered.

We included it in ```pypop-lso``` *just* for historical reason and research purpose (e.g., benchmarking). Of course, now it is still highly encouraged to continually check and improve the current implementation especially for large-scale black-box optimization.

Note that the common ```restart``` option has been added as default, in order to alleviate possible instability issue.

## Main Features

For ES, the self-adaptation of **individual** step-sizes (versus the *global* step-size) is generally crucial for obtaining high-precision results on ill-conditioned function landscapes. Interestingly, the currently very popular optimizer ```Adam``` can be roughly seen as its gradient-based counterpart but with often efficient performance for Deep Learning.

"*The concept of relatively large mutations within one generation, but passing only smaller variations to the next generation, is applicable successfully to parameter optimization superimposed with Gaussian noise.*"

## Numerical Experiments

We try our best to repeat numerical experiments of the original paper, though not always successful. See the easy-to-run [source code](https://github.com/os-popt/pypop-lso/blob/master/test/optimizers/es/ostermeier/repeat_experiments.py) for checking and/or improving them.

**Figure 1**

![Figure 1](https://raw.githubusercontent.com/os-popt/pypop-lso/master/test/optimizers/es/ostermeier/Ostermeier-Figure-1.png)

**Figure 3**

![Figure 3](https://raw.githubusercontent.com/os-popt/pypop-lso/master/test/optimizers/es/ostermeier/Ostermeier-Figure-3.png)

**Figure 4**

![Figure 4](https://raw.githubusercontent.com/os-popt/pypop-lso/master/test/optimizers/es/ostermeier/Ostermeier-Figure-4.png)

.......
