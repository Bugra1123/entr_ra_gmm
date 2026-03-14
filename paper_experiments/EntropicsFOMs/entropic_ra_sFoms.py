import EntropicFOMs
from EntropicFOMs import sFoms
import numpy as np
import pandas as pd

class entropic_ra_sFoms(sFoms):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)

    # Risk averse sFoms on quadratic functions
    def raGD_quad_params(self,trade_off,confidence=0.95):
        """
        Computes the  parameters of the risk averse gradient descend algorithm on strongly convex quadratic
        objective by solving [Can and Gurbuzbalaban, 2022. Problem (32)].
        :param:
            trade_off: float
                The trade off, i.e. $\varepsilon$ in (32), between the convergence rate
                and the evar of GD on quadratic objectives.

            confidence: float (default: 0.95)
                The confidence level, i.e. $\zeta$, at which EV@R of GD is computed.

        :return:
            ra_lr: np.array
                Learning rate of risk-averse GD
            ra_it_mom: np.array  (= 0)
                Momentum parameter of RA-GD at iterations, which is 0.
            ra_grad_mom: np.array (= 0)
                Momentum parameter of RA-GD at gradient, which is 0.
        """

        # Find the stable region
        L, mu = max(self.eig_vals), min(self.eig_vals)
        rho_opt=1-2/(np.sqrt(3*L/mu+1))
        stab_region = self.quad_gd_stab_reg(self.eig_vals)
        # stab_region=[alpha,beta,gamma, rho]
        const_reg_ind = np.where(stab_region[:, 3]**2 <= (1 + trade_off) * rho_opt**2)
        const_region = stab_region[const_reg_ind, :]
        evar = float("inf")
        ra_lr, ra_it_mom, ra_grad_mom = None, None, None

        for params in const_region[0]:
            a, b, g, _ = params
            # Use the Evar definition
            # evar_temp = self.quad_evar(a, b, g, self.eig_vals, variance=self.noise_std, conf_level=confidence)
            # Use the Evar bound for quadratics
            evar_temp = self.quad_evar_bound(a, b, g, self.eig_vals, std=self.noise_std, conf_level=confidence)

            if evar_temp < evar:
                evar = evar_temp
                ra_lr, ra_it_mom, ra_grad_mom = a, b, g
        return ra_lr, ra_it_mom, ra_grad_mom

    def raAGD_quad_params(self,trade_off,confidence=0.95):
        """
        Computes the  parameters of the risk averse gradient descend algorithm on strongly convex quadratic
        objective by solving [Can and Gurbuzbalaban, 2022. Problem (32)].
        :param:
            trade_off: float
                The trade off, i.e. $\varepsilon$ in (32), between the convergence rate
                and the evar of AGD on quadratic objectives.

            confidence: float (default: 0.95)
                The confidence level, $\zeta$, at which EV@R of AGD is computed.

        :return:
            ra_lr: np.array
                Learning rate of risk-averse AGD
            ra_it_mom: np.array
                Momentum parameter of RA-AGD at iterations.
            ra_grad_mom: np.array (= ra_it_mom)
                Momentum parameter of RA-AGD at gradient and equals to ra_it_mom.
        """
        # Find the stable region
        L, mu = max(self.eig_vals), min(self.eig_vals)
        rho_opt=1-2/(np.sqrt(3*L/mu+1))
        stab_region = self.quad_agd_stab_reg(self.eig_vals)
        # stab_region=[alpha,beta,gamma, rho]

        const_reg_ind = np.where(stab_region[:, 3]**2 <= (1 + trade_off) * rho_opt**2)
        const_region = stab_region[const_reg_ind, :]
        evar = float("inf")
        ra_lr, ra_it_mom, ra_grad_mom = None, None, None

        for params in const_region[0]:
            a, b, g, _ = params
            # Use the Evar definition
            # evar_temp = self.quad_evar(a, b, g, self.eig_vals, variance=self.noise_std, conf_level=confidence)

            # Use the Evar bound for quadratics
            evar_temp = self.quad_evar_bound(a, b, g, self.eig_vals, std=self.noise_std, conf_level=confidence)
            if evar_temp < evar:
                evar = evar_temp
                ra_lr, ra_it_mom, ra_grad_mom = a, b, g
        return ra_lr, ra_it_mom, ra_grad_mom

    def raTMM_quad_params(self, trade_off,confidence=0.95):
        """
        Computes the  parameters of the risk averse GMM (TMM with abuse of notation here) algorithm on
        strongly convex quadratic objective by solving [Can and Gurbuzbalaban, 2022. Problem (32)].
        :param:
            trade_off: float
                The trade off, i.e. $\varepsilon$ in (32), between the convergence rate and the evar of GMM on quadratic objectives.

            confidence: float (default: 0.95)
                The confidence level, $\zeta$, at which EV@R of GMM is computed.

        :return:
            ra_lr: np.array
                Learning rate of risk-averse GMM
            ra_it_mom: np.array
                Momentum parameter of RA-GMM at iterations.
            ra_grad_mom: np.array
                Momentum parameter of RA-GMM at gradient.
        """
        # Find the stable region
        L, mu = max(self.eig_vals), min(self.eig_vals)
        rho_opt=1-2/(np.sqrt(3*L/mu+1))
        stab_region= self.quad_stab_reg(self.eig_vals)
        # stab_region=[alpha,beta,gamma, rho]

        const_reg_ind= np.where(stab_region[:,3]**2<=(1+trade_off)*rho_opt**2)
        const_region=stab_region[const_reg_ind,:]
        evar=float("inf")
        ra_lr, ra_it_mom, ra_grad_mom = None, None, None

        for params in const_region[0]:
            a, b, g, _ = params
            # Use the Evar definition
            # evar_temp=self.quad_evar(a,b,g,self.eig_vals,variance=self.noise_std,conf_level=confidence)
            # Use the Evar bound for quadratics
            evar_temp = self.quad_evar_bound(a, b, g, self.eig_vals, std=self.noise_std, conf_level=confidence)

            if evar_temp< evar:
                evar=evar_temp
                ra_lr, ra_it_mom, ra_grad_mom = a, b, g
        return ra_lr,ra_it_mom, ra_grad_mom

    # Risk averse sFoms on strongly convex objectives
    def raTMM_str_cnv_params(self, L,mu, trade_off,dimension,confidence=0.95):
        """
        Computes the  parameters of the risk-averse generalized momentum methods on strongly convex
        non-quadratic objectives by solving [Can and Gurbuzbalaban, 2022. Problem (51)].
        :param:
            trade_off: float
                The trade off, i.e. $\varepsilon$ in (51), between the convergence rate and the evar of GMM
                on quadratic objectives.
            dimension: int
                The dimension of the problem, i.e. the number of features.
            confidence: float (default: 0.95)
                The confidence level, $\zeta$, at which EV@R of GMM is computed.

        :return:
            ra_lr: np.array
                Learning rate of risk-averse AGD
            ra_it_mom: np.array
                Momentum parameter of RA-AGD at iterations.
            ra_grad_mom: np.array
                Momentum parameter of RA-AGD at gradient and equals to ra_it_mom.
        """
        # Find the stable region
        rho2_opt = 1 - np.sqrt(mu / L)
        stab_region = self.str_cnvx_stab_reg_params(L,mu).T
        # stab_region=[vartheta,psi,rho2] where psi != 1
        const_reg_ind = np.where(stab_region[:, 2] <= (1 + trade_off) * rho2_opt)
        const_region = stab_region[const_reg_ind, :]
        evar = float("inf")
        ra_lr, ra_it_mom, ra_grad_mom = None, None, None

        for params in const_region[0]:
            varth, psi, _ =params
            # Use the Evar bound strongly convex functions
            evar_temp = self.str_cnvx_evar_bound(L,mu,varth,psi,dimension,confidence,self.noise_std)
            if evar_temp < evar:
                evar = evar_temp
                ra_lr= (1-varth)/(L*(1-psi))
                ra_it_mom=(1-np.sqrt(varth*ra_lr*mu))/(1-ra_lr*psi*mu)*(1-np.sqrt(ra_lr*mu/varth))
                ra_grad_mom=psi*ra_it_mom


        return ra_lr, ra_it_mom, ra_grad_mom

    def raAGD_str_cnv_params(self, L,mu, trade_off,dimension, confidence=0.95):
        """
       Computes the  parameters of the risk-averse accelerated gradient descend algorithm on strongly convex
       non-quadratic objectives by solving [Can and Gurbuzbalaban, 2022. Problem (51)].
       :param:
           trade_off: float
               The trade off, i.e. $\varepsilon$ in (51), between the convergence rate and the evar of GMM
               on quadratic objectives.
           dimension: int
               The dimension of the problem, i.e. the number of features.
           confidence: float (default: 0.95)
               The confidence level, $\zeta$, at which EV@R of GMM is computed.

       :return:
           ra_lr: np.array
               Learning rate of risk-averse AGD
           ra_it_mom: np.array
               Momentum parameter of RA-AGD at iterations.
           ra_grad_mom: np.array (= ra_it_mom)
               Momentum parameter of RA-AGD at gradient and equals to ra_it_mom.
       """
        # Find the stable region
        rho2_opt = 1 - np.sqrt(mu / L)
        stab_region = self.str_cnvx_agd_stab_reg_params(L,mu).T
        # stab_region=[alpha,rho2]
        const_reg_ind = np.where(stab_region[:, 1] <= (1 + trade_off) * rho2_opt)
        const_region = stab_region[const_reg_ind, :]
        evar = float("inf")
        ra_lr, ra_it_mom, ra_grad_mom = None, None, None

        for params in const_region[0]:
            alpha, _ =params
            # Use the Evar definition
            # evar_temp=self.quad_evar(a,b,g,self.eig_vals,variance=self.noise_std,conf_level=confidence)
            # Use the Evar bound for quadratics
            evar_temp = self.str_cnvx_agd_evar_bound(L,mu,alpha, dimension,confidence,self.noise_std)
            if evar_temp < evar:
                evar = evar_temp
                ra_lr= alpha
                ra_it_mom=(1-np.sqrt(alpha*mu))/(1+np.sqrt(alpha*mu))
                ra_grad_mom=ra_it_mom

        return ra_lr, ra_it_mom, ra_grad_mom

    def emp_risk_meas(self, subopt_data: np.ndarray,theta):
        """
        Calculates the empirical risk measure using the suboptimality path.
        ..math:
            \tilde{r}_{k,\sigma^2}(\theta)= \frac{2\sigma^2}{\theta}\log \sum_{i=1}^{N} e^{\frac{\theta}{2\sigma^2}(fk[i]-f*),
        where fk is the function value computed at k-th iteration on i-th sampled path.

        :param subopt_data: np.array (num_of_samples, num_of_iterations.)
            The suboptimality data containing suboptimality information, i.e. f(x_k^{(i)})-f(x_*(
        :param theta:
            The risk averseness parameter.
        :return:
            empr_risk: np.array (# of iterations)
                The empirical risk measure calculated at each iteration from the sample paths.
        """
        exponential=np.exp(0.5*theta/(self.noise_std**2)*subopt_data)
        return 2*(self.noise_std**2)/theta*np.log(exponential.mean(1))
