import math
import torch
from torch.optim import Optimizer
from .risk import RAGMMBounds


class GMM(Optimizer):
    """
    Generalized Momentum Method (GMM) optimizer.

    Implements the triple-momentum update rule:

    .. math::

        y_k       = x_k + \\gamma \\, d_k

        d_{k+1}   = \\beta \\, d_k - \\alpha \\, \\nabla f(y_k)

        x_{k+1}   = x_k + d_{k+1}

    Can be used standalone or wrapped by RAGMM, which selects
    $\\alpha$ (``lr``), $\\beta$ (``beta``), and $\\gamma$ (``gamma``) via
    an EV@R grid search.

    Args:
        params: iterable of parameters to optimize
        lr:     step size $\\alpha$ (default: 1e-3)
        beta:   momentum coefficient $\\beta \\in [0, 1)$ (default: 0.9)
        gamma:  lookahead coefficient $\\gamma$; 0 disables lookahead (default: 0.0)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, gamma=0.0):
        defaults = dict(lr=lr, beta=beta, gamma=gamma)
        super().__init__(params, defaults)

    def apply_lookahead(self):
        """Shift params to $y_k = x_k + \\gamma d_k$, saving $x_k$ for restoration in step()."""
        for group in self.param_groups:
            gamma = group['gamma']
            if gamma == 0.0:
                continue
            for p in group['params']:
                state = self.state[p]
                buf = state.get('momentum_buffer')
                if buf is None:
                    continue
                state['x_k_backup'] = p.data.clone()
                p.data.add_(buf, alpha=gamma)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one generalized momentum step.

        Without closure: uses gradients already in ``.grad`` (standard PyTorch pattern).
        With closure:    applies lookahead to $y_k$, evaluates the closure there,
                         restores $x_k$, then applies the update.
        """
        if closure is not None:
            self.apply_lookahead()
            with torch.enable_grad():
                loss = closure()
        else:
            loss = None

        for group in self.param_groups:
            lr   = group['lr']
            beta = group['beta']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'x_k_backup' in state:
                    p.data.copy_(state.pop('x_k_backup'))
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                d     = state['momentum_buffer']
                d_new = beta * d - lr * p.grad
                p.data.add_(d_new)
                state['momentum_buffer'] = d_new

        return loss


class RAGMM:
    """
    Risk-Averse Generalized Momentum Method (RA-GMM, Can et al. 2025).

    Selects ``lr``, ``beta``, ``gamma`` for a wrapped GMM optimizer by minimising an
    EV@R convergence bound via grid search over $(\\alpha, \\psi)$, given exact smoothness
    $L$ and strong-convexity $\\mu$.  The grid search runs once on the first step
    (when the parameter dimension $d$ is known) and the result is fixed.

    $L$ and $\\mu$ must be supplied by the caller (e.g. computed analytically for
    logistic regression on fixed features).

    Args:
        optimizer:    GMM instance to control
        L:            gradient-Lipschitz constant $L$ of the objective
        mu:           strong-convexity constant $\\mu$ of the objective
        zeta:         EV@R confidence level $\\zeta \\in (0, 1)$ (default: 0.05)
        grid_size:    grid points per axis in the $(\\alpha, \\psi)$ search (default: 10)
        eps:          numerical floor (default: 1e-8)
        alpha_range:  ``(lo, hi)`` multipliers on $\\alpha_\\text{init}$ defining the search range;
                      ``lo`` scales the lower bound ($\\alpha_\\text{lo} = \\text{lo} \\cdot \\alpha_\\text{init}$),
                      ``hi`` scales the upper bound ($\\alpha_\\text{hi} = (1+\\text{hi}) \\cdot \\alpha_\\text{init}$, capped at $1/L$)
        psi_range:    ``(lo, hi)`` floor and ceiling for the $\\psi$ search (default: (1e-5, 0.95))
        rate_slack:   max allowed $\\rho$ inflation above the GD-optimal rate (default: 0.05)

    Usage::

        gmm = GMM(model.parameters(), lr=1/(L+mu))
        opt = RAGMM(gmm, L=L, mu=mu, zeta=0.05)

        # Training loop
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        opt.step()
    """

    def __init__(
        self,
        optimizer,
        L,
        mu,
        zeta=0.05,
        grid_size=10,
        eps=1e-8,
        alpha_range=(1e-6, 1e-1),
        psi_range=(1e-5, 0.95),
        rate_slack=0.05,
    ):
        if not isinstance(optimizer, GMM):
            raise TypeError(
                f"optimizer must be a GMM instance, got {type(optimizer).__name__}"
            )

        self.optimizer   = optimizer
        self.L           = float(L)
        self.mu          = float(mu)
        self.zeta        = zeta
        self.grid_size   = grid_size
        self.eps         = eps
        self.alpha_range = alpha_range
        self.psi_range   = psi_range
        self.rate_slack  = rate_slack

        self._alpha_init = optimizer.param_groups[0]['lr']

        self.bounds = RAGMMBounds(zeta=zeta, eps=eps)

        self.dimension  = None
        self._evar_bound = float('inf')
        self._tuned      = False
        self.step_count  = 0

    # ------------------------------------------------------------------ #
    # Convenience proxies                                                  #
    # ------------------------------------------------------------------ #

    @property
    def alpha(self):
        """Current step size $\\alpha$ (``lr`` of the wrapped GMM)."""
        return self.optimizer.param_groups[0]['lr']

    @property
    def beta(self):
        """Current momentum coefficient $\\beta$."""
        return self.optimizer.param_groups[0]['beta']

    @property
    def gamma(self):
        """Current lookahead coefficient $\\gamma$."""
        return self.optimizer.param_groups[0]['gamma']

    def zero_grad(self):
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_grid_search(self):
        """Grid search over $(\\alpha, \\psi)$ to minimise EV@R bound subject to convergence rate constraint."""
        L_val  = self.L
        mu_val = self.mu

        alpha_lo = max(self._alpha_init * self.alpha_range[0], self.eps)
        alpha_hi = min(self._alpha_init * (1 + self.alpha_range[1]), 1 / L_val)

        # Log-uniform spacing: covers multiple orders of magnitude evenly
        alpha_vals = torch.exp(
            torch.linspace(math.log(alpha_lo), math.log(alpha_hi), steps=self.grid_size)
        )

        best_alpha = self._alpha_init
        best_beta  = 0.0
        best_gamma = 0.0
        best_psi   = 0.0
        best_bound = float('inf')

        for alpha in alpha_vals:
            alpha_val     = alpha.item()
            rho_benchmark = 1 - math.sqrt(alpha_val * mu_val / L_val)

            # $\psi$ feasibility bounds (derived from is_feasible conditions):
            #   $\vartheta > 0$          =>  $\psi > 1 - 1/(\alpha L)$
            #   $\alpha \psi L < 1$      =>  $\psi < 1/(\alpha L)$
            psi_lo = max(self.eps, 1.0 - 1.0 / (alpha_val * L_val))
            psi_hi = 1.0
            if psi_lo >= psi_hi:
                continue

            for psi in torch.linspace(psi_lo, psi_hi, steps=self.grid_size):
                psi_val = psi.item()
                if not self.bounds.is_feasible(alpha_val, psi_val, L_val, mu_val):
                    continue

                beta_val  = float(self.bounds._beta(alpha_val, psi_val, L_val, mu_val))
                gamma_val = float(self.bounds._gamma(beta_val, psi_val))

                # Stability guard: $\beta \geq 1$ makes the momentum update diverge.
                if not (0.0 <= beta_val < 1.0) or not (0.0 <= gamma_val < 1.0):
                    continue

                vartheta = self.bounds._vartheta(alpha_val, psi_val, L_val)
                rho_val  = 1.0 - math.sqrt(vartheta * alpha_val * mu_val)
                if not (0.0 < rho_val < 1.0):
                    continue

                bound = self.bounds.evar_bound(alpha_val, psi_val, L_val, mu_val)

                if bound < best_bound and rho_val / rho_benchmark < (1 + self.rate_slack):
                    best_alpha = alpha_val
                    best_beta  = beta_val
                    best_gamma = gamma_val
                    best_psi   = psi_val
                    best_bound = bound

        return best_alpha, best_beta, best_gamma, best_psi, best_bound

    def _apply_hyperparams(self, alpha, beta, gamma):
        for group in self.optimizer.param_groups:
            group['lr']    = alpha
            group['beta']  = beta
            group['gamma'] = gamma

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def step(self, closure=None):
        """
        Tune GMM hyperparameters (once) and perform one update step.

        Flow:
            1. Apply lookahead with current $\\gamma$ and evaluate gradient (closure path),
               or use gradients already in ``.grad`` (no-closure path).
            2. On the first step (once $d$ is known): run grid search → set GMM's $\\alpha$/$\\beta$/$\\gamma$.
            3. Delegate the parameter update to the wrapped GMM.

        Returns a dict of diagnostic statistics.
        """
        self.step_count += 1

        if closure is not None:
            self.optimizer.apply_lookahead()
            with torch.enable_grad():
                closure()

        # Collect flattened gradients for dimension initialisation and diagnostics
        grads = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grads.append(p.grad.data.view(-1))
        if not grads:
            return {}

        g = torch.cat(grads)
        if self.dimension is None:
            self.dimension  = g.numel()
            self.bounds.d   = self.dimension

        sigma_hat = float(g.std().clamp(min=self.eps))

        tuned = False
        if not self._tuned:
            alpha, beta, gamma, psi, bound = self._run_grid_search()
            self._apply_hyperparams(alpha, beta, gamma)
            self._evar_bound = bound
            self._tuned      = True
            tuned            = True
        else:
            alpha = self.alpha
            beta  = self.beta
            gamma = self.gamma
            psi   = 0.0
            bound = self._evar_bound

        # Backup params for NaN/Inf recovery
        param_backups = {
            p: p.data.clone()
            for group in self.optimizer.param_groups
            for p in group['params'] if p.grad is not None
        }

        self.optimizer.step()

        nan_detected = False
        for p, backup in param_backups.items():
            if not torch.isfinite(p.data).all():
                nan_detected = True
                p.data.copy_(backup)
                p_state = self.optimizer.state.get(p, {})
                if 'momentum_buffer' in p_state:
                    p_state['momentum_buffer'].zero_()

        return {
            'alpha':        alpha,
            'beta':         beta,
            'gamma':        gamma,
            'psi':          psi,
            'sigma_hat':    sigma_hat,
            'evar_bound':   bound,
            'step':         self.step_count,
            'tuned':        tuned,
            'nan_detected': nan_detected,
        }
