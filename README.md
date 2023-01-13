# First-Order Probabilistic Programming Language (FOPPL)

This repository contains a fully working [FOPPL](https://arxiv.org/abs/1809.10756) interpreter. A FOPPL program is one in which it is possible to convert the program into a finite graphical model, and conversely any statistical problem that can be written as a finite graphical model can be written in FOPPL. This code can ingest either the graph-based or evaluation-based outputs of [daphne](https://github.com/plai-group/daphne) and run a number of different inference algorithms to perform Bayesian posterior inference in *any* first-order probabilistic program. If your problem can be written in FOPPL (and therefore written as a graph) then this repository can perform Bayesian inference.

The code uses [pytorch](https://pytorch.org/) primitives, and thus supports automatic differentiation. This ensures that inference algorithms that require derivatives (e.g., Hamiltonian Monte Carlo; Variational inference) are supported.

Inference algorithms:
- Importance sampling
- Metropolis-Hastings Monte Carlo
- Gibbs Monte Carlo
- Hamiltonian Monte Carlo
- Variational inference

In future we would like to include the [inference compilation](https://arxiv.org/abs/1610.09900) algorithm within the context of graphical models. It should be possible to use the reverse graphical model to limit the number of neural networks required as surrogate link functions.

To run, simply type

```
python run.py
```

edit the `config.yaml` (which uses [hydra](https://hydra.cc/docs/intro/)) to choose which probablistic programs (and tests) to run.

Dependencies (`pip install numpy` etc.):

```
numpy
pytorch
hydra
wandb
ttvfast
```
