# Genitor

## Reference

```
Whitley, D., Dominic, S., Das, R. and Anderson, C.W., 1993.
Genetic reinforcement learning for neurocontrol problems.
Machine Learning, 13(2-3), pp.259-284.
https://link.springer.com/content/pdf/10.1007/BF00993045.pdf

Moriarty, D.E., Schultz, A.C. and Grefenstette, J.J., 1999.
Evolutionary algorithms for reinforcement learning.
Journal of Artificial Intelligence Research, 11, pp.241-276.
https://www.jair.org/index.php/jair/article/view/10240/24373

Such, F.P., Madhavan, V., Conti, E., Lehman, J., Stanley, K.O. and Clune, J., 2017.
Deep neuroevolution: Genetic algorithms are a competitive alternative for training deep neural networks for reinforcement learning.
arXiv preprint arXiv:1712.06567.
https://arxiv.org/pdf/1712.06567.pdf
```

## Open-Source Implementation

In the library ```pypop-lso```, the source code of **Genetic Reinforcement Learning** is available at the GA class [GENITOR](https://github.com/os-popt/pypop-lso/blob/master/optimizers/ga/genitor.py).

We included it in ```pypop-lso``` *just* for historical reason and research purpose (e.g., benchmarking).

## Main Features

GENITOR is a steady-state genetic algorithm with a real-valued representation, a very high mutation rate, and unusually small populations.

Historically, GAs often involve crossover, but for simplicity this code did NOT include it. Note that the classical crossover operator has been abandoned by almost all modern GAs (e.g., [DeepGA](https://arxiv.org/pdf/1712.06567.pdf)) for large-scale numerical optimization, though still common for combinatorial optimization.

## Numerical Experiments

.......

