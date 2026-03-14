import math
import warnings



class RAGMMBounds:
    """
    EV@R convergence bounds for the RA-GMM algorithm (Can et al. 2025).

    Computes theoretical upper bounds on EV@R given exact smoothness $L$,
    strong-convexity $\\mu$, and parameter dimension $d$.  All methods are
    deterministic — no data or gradient samples are used.

    Args:
        zeta:  EV@R confidence level $\\zeta \\in (0, 1)$ (default: 0.05)
        eps:   numerical floor for intermediate computations (default: 1e-8)
    """

    def __init__(self, zeta=0.05, eps=1e-8):
        self.zeta = zeta
        self.eps = eps
        self.d = None  # set by RAGMM on first step

    # ------------------------------------------------------------------ #
    # Feasibility check                                                    #
    # ------------------------------------------------------------------ #

    def is_feasible(self, alpha, psi, L, mu):
        """
        Return True iff $(\\alpha, \\psi)$ satisfy the preconditions for all
        bound computations to be well-defined:

        - $\\vartheta = 1 - \\alpha L (1 - \\psi) > 0$
        - $\\alpha \\psi L < 1$  (avoids division by zero in $\\beta$)
        - $\\vartheta \\alpha \\mu \\in (0, 1)$  (convergence rate strictly in $(0, 1)$)
        """
        vartheta = 1 - alpha * L * (1 - psi)
        if vartheta <= 0:
            return False
        if alpha * psi * L >= 1:
            return False
        prod = vartheta * alpha * mu
        if prod <= 0 or prod >= 1:
            return False
        return True

    # ------------------------------------------------------------------ #
    # Parameter formulas (Theorem constants)                              #
    # ------------------------------------------------------------------ #

    def _vartheta(self, alpha, psi, L):
        """$\\vartheta = 1 - \\alpha L (1 - \\psi)$, clamped away from zero."""
        return max(1 - alpha * L * (1 - psi), 1e-8)

    def _beta(self, alpha, psi, L, mu):
        """Optimal momentum coefficient $\\beta$ for given $(\\alpha, \\psi)$."""
        vartheta = self._vartheta(alpha, psi, L)
        m1 = (1 - math.sqrt(vartheta * alpha * mu)) / (1 - alpha * psi * L)
        m2 = 1 - math.sqrt(alpha * mu / vartheta)
        return m1 * m2

    def _gamma(self, beta, psi):
        """Lookahead coefficient $\\gamma = \\psi \\beta$."""
        return psi * beta

    def _lyapunov_coeff(self, alpha, psi, L, mu):
        """Lyapunov constant $V$ from the convergence analysis."""
        beta    = self._beta(alpha, psi, L, mu)
        gamma   = self._gamma(beta, psi)
        vartheta = self._vartheta(alpha, psi, L)

        term_1 = 2 * (beta - gamma) ** 2
        term_2 = (1 - alpha * L) ** 2 * (1 + 2 * gamma + 2 * gamma ** 2)
        term_3 = vartheta / (2 * alpha) * (1 - math.sqrt(vartheta * alpha * mu))
        return 2 * (L ** 2) / mu * (term_1 + term_2) + term_3

    def _theta_max(self, alpha, psi, L, mu):
        """Upper bound on the EV@R dual variable $\\theta$."""
        vartheta = self._vartheta(alpha, psi, L)
        V        = self._lyapunov_coeff(alpha, psi, L, mu)
        numerator   = 2 * math.sqrt(vartheta * mu)
        denominator = alpha * (
            8 * V * math.sqrt(alpha)
            + math.sqrt(vartheta * mu) * (vartheta + alpha * L)
        )
        theta_ub = numerator / denominator
        if theta_ub <= 0:
            raise ValueError("theta upper bound must be positive")
        return theta_ub

    # ------------------------------------------------------------------ #
    # Convergence and confidence bounds                                    #
    # ------------------------------------------------------------------ #

    def _rho_bound(self, alpha, psi, L, mu):
        """Upper bound on the convergence rate $\\rho$."""
        vartheta = self._vartheta(alpha, psi, L)
        V        = self._lyapunov_coeff(alpha, psi, L, mu)
        theta_ub = self._theta_max(alpha, psi, L, mu)

        denom = 2 - theta_ub * alpha * (vartheta + alpha * L)
        if denom <= 0:
            return float('inf')
        common = (4 * theta_ub * (alpha ** 2) * V) / denom
        term   = 1 - math.sqrt(vartheta * alpha * mu) + common
        disc   = term ** 2 + 4 * common
        if disc < 0:
            return float('inf')
        rate = 0.5 * term + 0.5 * math.sqrt(disc)
        if rate < 0 or rate >= 1:
            return float('inf')
        return rate

    def _confidence_bound(self, alpha, psi, L, mu):
        """Threshold that determines the high- / low-confidence regime."""
        vartheta = self._vartheta(alpha, psi, L)
        theta_ub = self._theta_max(alpha, psi, L, mu)
        rho      = self._rho_bound(alpha, psi, L, mu)

        if not math.isfinite(rho) or rho >= 1.0:
            return float('inf')
        d = self.d if self.d is not None else 1
        denom_m2 = 2 - theta_ub * alpha * (vartheta + alpha * L)
        if denom_m2 <= 0:
            return float('inf')

        m1 = 0.5 * d / (1.0 - rho)
        m2 = (theta_ub * alpha * (vartheta + alpha * L)) / denom_m2
        return m1 * (m2 ** 2)

    # ------------------------------------------------------------------ #
    # Main bound                                                           #
    # ------------------------------------------------------------------ #

    def evar_bound(self, alpha, psi, L, mu, sigma=1.0, zeta=None):
        """
        EV@R upper bound for RA-GMM at hyperparameters $(\\alpha, \\psi)$.

        Args:
            alpha:  step size $\\alpha$
            psi:    reparametrisation variable $\\psi$
            L:      gradient-Lipschitz constant $L$
            mu:     strong-convexity constant $\\mu$
            sigma:  gradient noise level $\\sigma$ (scales the bound as $\\sigma^2$;
                    pass 1.0 to use the bound as a dimensionless ranking
                    metric for the grid search)
            zeta:   confidence level $\\zeta$; defaults to ``self.zeta``
        """
        if zeta is None:
            zeta = self.zeta

        vartheta = self._vartheta(alpha, psi, L)
        rho = self._rho_bound(alpha, psi, L, mu)
        if math.isinf(rho) or rho >= 1.0:
            return float('inf')

        d           = self.d if self.d is not None else 1
        rate_factor = 1 - rho

        try:
            c_bound = self._confidence_bound(alpha, psi, L, mu)
            if math.log(1 / zeta) <= c_bound:
                # High-confidence regime (eq. 42)
                m1 = alpha * (vartheta + alpha * L)
                m2 = math.sqrt(d / rate_factor) + math.sqrt(2 * math.log(1 / zeta))
                return 0.5 * (sigma ** 2) * m1 * (m2 ** 2)
            else:
                # Low-confidence regime (eq. 43)
                theta_ub = self._theta_max(alpha, psi, L, mu)
                denom2 = 2 - theta_ub * alpha * (vartheta + alpha * L)
                if denom2 <= 0:
                    return float('inf')
                t1 = d * alpha * (vartheta + alpha * L) / (rate_factor * denom2)
                t2 = 2 * math.log(1 / zeta) / theta_ub
                return sigma ** 2 * (t1 + t2)

        except (ValueError, AssertionError) as e:
            warnings.warn(f"EV@R bound computation failed: {e}")
            return float('inf')
