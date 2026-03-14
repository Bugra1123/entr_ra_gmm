# Risk-Averse Tuning for Momentum Methods

PyTorch implementation of the **RA-GMM** (Risk-Averse Generalized Momentum Method) optimizer from:

> **Entropic Risk-Averse Generalized Momentum Methods**
> Bugra Can, Mert Gürbüzbalaban
> *Optimization Methods and Software*, 40(6), 1535–1583, 2025
> https://doi.org/10.1080/10556788.2025.2549356

## Overview

This package provides a PyTorch optimizer that systematically selects momentum hyperparameters by minimising an **Entropic Value at Risk (EV@R)** convergence bound, rather than relying on fixed or hand-tuned parameters.

The Generalized Momentum Method (GMM) covers a broad class of first-order algorithms as special cases — including gradient descent (GD), Nesterov's accelerated gradient descent (AGD), and the heavy-ball (HB) method — via three parameters $\alpha$ (step size), $\beta$ (momentum), and $\gamma$ (lookahead):

$$y_k = x_k + \gamma \, d_k$$

$$d_{k+1} = \beta \, d_k - \alpha \, \nabla f(y_k)$$

$$x_{k+1} = x_k + d_{k+1}$$

**RA-GMM** wraps a `GMM` optimizer and, on the first step, runs a grid search over $(\alpha, \psi)$ to find the hyperparameters that minimise the theoretical EV@R upper bound subject to a convergence rate constraint.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import torch
from risk_averse_tuning import GMM, RAGMM

# L = gradient-Lipschitz constant, mu = strong-convexity constant
L, mu = 10.0, 0.1

gmm = GMM(model.parameters(), lr=1 / (L + mu))
opt = RAGMM(gmm, L=L, mu=mu, zeta=0.05)

for x, y in dataloader:
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    stats = opt.step()
```

`opt.step()` returns a diagnostics dict: `alpha`, `beta`, `gamma`, `psi`, `evar_bound`, `sigma_hat`, `step`, `tuned`, `nan_detected`.

## Components

| Module | Description |
|---|---|
| `optimizer.py` | `GMM` — PyTorch `Optimizer` implementing the triple-momentum update. `RAGMM` — wrapper that tunes GMM via EV@R grid search. |
| `risk.py` | `RAGMMBounds` — computes theoretical EV@R upper bounds from Theorem 1/3/5 of the paper. |

## RAGMM Parameters

| Parameter | Default | Description |
|---|---|---|
| `L` | required | Gradient-Lipschitz constant $L$ |
| `mu` | required | Strong-convexity constant $\mu$ |
| `zeta` | `0.05` | EV@R confidence level $\zeta \in (0, 1)$ |
| `grid_size` | `10` | Grid points per axis in the $(\alpha, \psi)$ search |
| `alpha_range` | `(1e-6, 1e-1)` | Multipliers on $\alpha_\text{init}$ defining the search range |
| `psi_range` | `(1e-5, 0.95)` | Floor and ceiling for the $\psi$ search |
| `rate_slack` | `0.05` | Max allowed $\rho$ inflation above GD-optimal rate |

## Reference

```bibtex
@article{can2025entropic,
  title   = {Entropic risk-averse generalized momentum methods},
  author  = {Can, Bugra and G\"{u}rb\"{u}zbalaban, Mert},
  journal = {Optimization Methods and Software},
  volume  = {40},
  number  = {6},
  pages   = {1535--1583},
  year    = {2025},
  doi     = {10.1080/10556788.2025.2549356}
}
```
