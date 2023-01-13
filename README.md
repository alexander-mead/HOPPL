# Higher-Order Probabilistic Programming Language (HOPPL)

This repository contains a fully working [HOPPL](https://arxiv.org/abs/1809.10756) interpreter. A HOPPL program is one in which it is not possible to convert the program into a finite graphical model, this occurs when probabilistic programs (statistical models) are capable of creating random numbers of random variables at each instantiation. This means that the statistical model cannot be represented by a finite graphical model, and this limits the number of algorithms that can be used to perform Bayesian inference. This code ingests the evaluation-based outputs of [daphne](https://github.com/plai-group/daphne) and then can run a number of different inference algorithms to perform Bayesian posterior inference in *any* higher-order probabilistic program. If your problem can be written in HOPPL then this repository can perform Bayesian inference.

The code uses [pytorch](https://pytorch.org/) primitives, and thus supports automatic differentiation. This ensures that inference algorithms that require derivatives are supported.

Inference algorithms:
- Importance sampling
- Sequential Monte Carlo

In future we would like to include the [light-weight Metropolis Hastings](https://web.stanford.edu/~ngoodman/papers/lightweight-mcmc-aistats2011.pdf) and  [inference compilation](https://arxiv.org/abs/1610.09900) algorithms within the context of a HOPPL.
