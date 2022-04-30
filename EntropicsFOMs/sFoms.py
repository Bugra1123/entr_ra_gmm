import numpy
import numpy as np
import logging, os
import pandas as pd
import plotly.graph_objects as go
from tabulate import tabulate
import plotly.express as px
# import EntropicFOMs
from EntropicFOMs.SyntheticData import *
from EntropicFOMs.ObjFunctions import *
import numpy.random as nprand
import numpy.linalg as nplin
logging.basicConfig(level = logging.INFO)

class sFoms(object):
    """
    The class for first order stochastic methods to solve the strongly convex optimization problem.

    Attributes
    ----------
        path:  str
            The path to the folder to which the results are going to be saved.
        optim_problem: str
            quadratic:          Smooth strongly convex quadratic cost function.
            synthetic_logit:    Smooth strongly convex logistic cost defined on synthetic data.
        init_x0:
            The initial point for the algorithm.
              None:     The initialization sampled from a standard uniform distribution.
              "fixed":  The initialization is unit vector.
              np.ndarray: (# of feat, 1)
        eig_vals: np.array
            The eigenvalues of the Hessian of quadratic objective function.
            Required by "quadratic" optimization problem..
        reg_param: np.array
            The regularization parameter of the optimization problem. Set to be 1 by default.
        noise_std: float (default: 0)
            The standard deviation of the additive Gaussian noise on the gradient of the cost function.
        data_std: float
            The standard deviation of the random matrix used to generate synthetic data.
            Required by "synthetic_logit" optimization problem.
        data_size:
            The size of the syntethic data required by "synthetic_logit" optimization problem.
        num_of_feat:
            The number of features in the "synthetic_logit" optimization problem.

    Functions
    -------
    train(learning_rate=0.01, iter_momentum=0, grad_momentum=0, stop_criteria=None, Samples=1, seed=None)
        Trains the model using stochastic first order method:
        .. math ::

            x_{k+1}=x_{k}+iter_momentum*(x_{k}-x_{k-1})-learning_rate*\tilde{\nabla}f(x_k)+\gamma*(x_{k}-x_{k-1})

    subopt_fig_data():
        Generates mean, one standard deviation below and above of the mean for the iterates of the  stochastic first order method.
    """
    # Path to the folder in which the results are going to be stored.
    # Enter local folder manually:
    # path = "local folder/",
    # or retrieve the current path and create an experiment results folder inside the current path:
    path=os.getcwd()+"/risk-averse GMM experiments/"
    # Check whether the path exists. Create if it does not
    if not os.path.isdir(path):
        os.mkdir(path)

    def __init__(self, optim_problem=None,eig_vals=None, init_x0=None, reg_param=0,
    noise_std=0, data_std=None, data_size=None, num_of_feat=None):
        self.__dict__.update(locals())
        # Quadratic objectives
        if optim_problem == "quadratic":
            if self.eig_vals is None:
                raise Exception("Enter the attribute: eig_vals ")
            self.num_of_feat = len(self.eig_vals)
            self.data_optim=None

            # Define the objective function
            self.func= lambda x: quadratic_cost(x, eigs=self.eig_vals, regulizer=self.reg_param, sigma=self.noise_std)

        # Logistic regression on synthetic data
        elif optim_problem == "synthetic_logit":
            # Check for attributes
            if (self.data_std or self.data_size or self.num_of_feat) is None:
                raise Exception("Enter the attributes: data_std, data_size and num_of_feat ")

            # Generate the synthetic data
            self.X, self.Y, self.data_optim = Create_Classification_Data(self.data_std, self.data_size, self.num_of_feat,
                                                                  seed=1)
            try:
                self.noise_std
            except:
                self.noise_std=0
            try:
                self.reg_param
            except:
                logging.info("Setting regularization parameter = 1")
                self.reg_param=float(1)
            # Define the objective function
            self.func = lambda w: logistic_cost(w, self.X, self.Y, reg=self.reg_param, sigma=self.noise_std)

        else:
            raise Exception("Unknown optimization problem. The allowed methods are 'quadratic' and 'synthetic_logit'.")
        
        if self.init_x0 is None and self.optim_problem is not None:
            self.init_x0 = nprand.normal(size=(self.num_of_feat, 1))
        elif self.init_x0 == "fixed" and self.optim_problem is not None:
            self.init_x0 = np.ones((self.num_of_feat, 1))

    def train(self, learning_rate=0.01, iter_momentum=0, grad_momentum=0, stop_criteria=None, Samples=1,seed=None):
        """
        Runs the first order methods under given choice of parameters to solve the optimization problem, optim_problem, until stopping criteria is satisfied.

        :param learning_rate: float
            The learning rate of the algorithm.
        :param iter_momentum: float (default: 0)
            The momentum parameter on the iterations, i.e. \beta
        :param grad_momentum: float (default: 0)
            The momentum parameter on the iterates on which the gradient is computed, i.e. \gamma
        :param stop_criteria: (default: None)
            None: The algorithm runs until the gradient norm computed at the iteration is below 1e-4 or number
                  of iterations is less than 2000.
            int: The algorithm runs until the number of iterations reaches stop_criteria.
        :param Samples: (default: 100)
            The number of sample paths that will be generated.
        :param seed: int (default: None)
            Fix the seed for random number generators.
        :return:
            csv file:
                The function values, generated using stochastic first
                order methods, together with the last iterate are saved to
                    "path/optim_problem".
        """
        if seed is not None: nprand.seed(seed)

        logging.info( f"Training model...\n"\
                      f"learning rate |  { learning_rate :2.3f} | Iteration momentum |  {iter_momentum :2.3f} "\
                      f"| Gradient Momentum | {grad_momentum : 2.3f}")
        # Store the results at fk_values.
        colnames=[str(i) for i in range(0,Samples)]
        self.suboptimality_data = pd.DataFrame([], columns=colnames)

        # Generate the results path
        if iter_momentum == 0 and grad_momentum == 0:
            method_name = "_GD" + "_Var_" + str(self.noise_std)
        elif iter_momentum != 0 and grad_momentum == 0:
            method_name = "_Polyak" + "_Var_" + str(self.noise_std)
        elif iter_momentum != 0 and grad_momentum == iter_momentum:
            method_name = "_AGD" + "_Var_" + str(self.noise_std)
        else:
            method_name = "_TMM" + "_Var_" + str(self.noise_std)

        # Assign results to attribute for further reference.
        self.suboptimality_data_path = self.path + self.optim_problem +"_subopt" + method_name + ".csv"
        self.log_path = self.path+self.optim_problem+"_log"+method_name+".txt"

        self.train_result=None # Collect last iterates of each trial

        # Generate the log for the algorithm
        alg_log=f"The Parameters: \n"\
                      f"learning rate |  { learning_rate :2.3f} | Iteration momentum |  {iter_momentum :2.3f} "\
                      f"| Gradient Momentum | {grad_momentum : 2.3f}"+"\n"
        x0=self.init_x0

        for trial in range(Samples):
            # Initiate algorithm
            run_algorithm = True
            num_of_iterations=1
            xk=x0
            xk_1=x0
            fn, gn = self.func(xk)
            iterates=[fn]
            if trial==0:
                logging.info(f"Iterations of 1st sample:")
                alg_log+=f"Iterations of 1st sample: \n"

            # Run the algorithm
            while run_algorithm:
                yk=xk+iter_momentum*(xk-xk_1)
                zk=xk+grad_momentum*(xk-xk_1)
                _, gn= self.func(zk)
                x_next=yk-learning_rate*gn
                xk_1=xk
                xk=x_next
                fn, gn = self.func(xk)
                iterates=np.append(np.around(iterates,decimals=5),fn)
                # Logging
                if trial == 0 and num_of_iterations%50==0 :
                    logging.info(f"Iteration: {num_of_iterations}| Function Value: {fn :.4f}"+f"| Norm of gradient: {np.linalg.norm(gn): .4f}")
                    alg_log+= f"Iteration: {num_of_iterations}| Function Value: {fn :.4f}"+f"| Norm of gradient: {np.linalg.norm(gn): .4f}"+"\n"
                # Stopping criteria
                num_of_iterations += 1
                if stop_criteria == None:
                    if nplin.norm(gn) < 1e-4 or num_of_iterations>=2000:
                        run_algorithm=False
                else:
                    if num_of_iterations > stop_criteria:
                        run_algorithm=False

            # Collect the iterations of trial.
            self.suboptimality_data[str(trial)]=iterates

            # Collect the last iterate of each trial
            if self.train_result is None:
                self.train_result=xk.reshape(-1, 1)
            else:
                self.train_result=np.append(self.train_result,xk.reshape(-1,1),axis=1)

        # Save data into csv file.
        self.suboptimality_data.to_csv(self.suboptimality_data_path)
        logging.info(f'Saved the suboptimality results into the path: \n' \
                     f'{self.suboptimality_data_path}')
        with open(self.log_path, 'w') as logfile:
            print(alg_log,file=logfile)

    def subopt_fig_data(self):
        """
        Generates mean, one standard deviation below and above of the mean for the iterates of the  stochastic first order method.
        :return:
        std_below: pandas.Series (# of iterations)
            The expected suboptimality - std of suboptimality at each iteration.
        mean:     pandas.Series (# of iterations)
            The expected suboptimality path generated by train().

        std_above: pandas.Series (# of iterations)
            The expected suboptimality + std of suboptimality at each iteration.
        """
        # Collect graph data
        graph_data=pd.read_csv(self.suboptimality_data_path, index_col=0)
        # Compute the mean and std
        mean, std = graph_data.mean(1), graph_data.std(1)
        std_above, std_below = mean + std, mean - std
        return std_below, mean, std_above

    def L_and_mu_logistic_reg(self):
        """
        Compute the upper bound on the L-smoothness parameter of the logistic regression over data.
        Source: http://lcsl.mit.edu/courses/isml2/isml2-2015/scribe14A.pdf
        :return:
        """
        s,f=self.X.shape
        eigs, _ =nplin.eig(np.matmul(self.X.T,self.X))
        return 1/s*eigs.max()+self.reg_param, self.reg_param

    def create_subopt_plot(self ,name, fig_color="rgba(0, 91, 247)", Figure= None):
        """
        Generates the plot of suboptimality data generated by train() function.
        :param name: str
            The name of the figure that s going to be printed in the legend
        :param fig_color: str (rgba color format, default: "rgba(0, 91, 247)")
            The color of the figure that is plotted.
        :param Figure: (default:  None)
            None:         Creates a new go.Figure() file and generates the suboptimality figure.
            go.Figure():  Generates the suboptimality figure and adds it into the go.Figure() file provided.
        :return:
            go.Figure()
        """

        if Figure == None:
            subopt_figures= go.Figure()
        else:
            subopt_figures= Figure
        std_below, mean, std_above = self.subopt_fig_data()
        subopt_figures.add_trace(go.Scatter(y=mean,
                                            name=name + "-mean",
                                            line=dict(width=5, color=fig_color[0:-1] + ",1)")))

        subopt_figures.add_trace(go.Scatter(y=std_above,
                                            name="",
                                            fill=None,
                                            # fillcolor='rgba(0, 91, 247,0.30)',
                                            line=dict(width=0, color=fig_color[0:-1] + ",0.25)"),
                                            showlegend=False))
        subopt_figures.add_trace(go.Scatter(y=std_below,
                                            name="",
                                            fill="tonexty",
                                            fillcolor=fig_color[0:-1] + ",0.25)",
                                            mode='none',
                                            showlegend=False))
        subopt_figures.update_layout(legend=dict(orientation="h",
                                                 xanchor="center",
                                                 x=0.5,
                                                 yanchor="top",
                                                 y=1,
                                                 ),
                                     font=dict(size=20),
                                     legend_font_size=25)
        return subopt_figures

    # Functions parameter selection and plotting regions
    #  Quadratic objectives
    def quad_stab_reg(self, eigs, nbins=50):
        """
        Calculates the stable region of GMM, S_q, given in [Can and Gurbuzbalaban, 2022] for convex quadratic
        objectives using grid search.
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param nbins: int
            The number of bins that is going to be used in grid search on all alpha, beta, and gamma.
        :return:
            region: np.array (alpha,beta,gamma ,rho)
                The list that contains the parameters alpha, beta, gamma belonging to stable set and the rate
                rho suggested by Lemma 3.1
        """
        beta, gamma = np.linspace(0,1,nbins), np.linspace(0,1,nbins)
        alpha= np.linspace(0,1/max(eigs),nbins)
        # Alpha should be positive
        alpha= np.delete(alpha, np.where(alpha==0))
        # check the type of eigs:
        if type(eigs) != numpy.ndarray:
            eigs=np.array(eigs)
        region=None
        for a in alpha:
            for b in beta:
                for g in gamma:
                    c_i = (1 + b) - a * (1 + g)*eigs
                    d_i = -b+a*g*eigs
                    rho_i_1=abs(c_i-np.emath.sqrt(c_i**2+4*d_i))/2
                    rho_i_2=abs(c_i+np.emath.sqrt(c_i**2+4*d_i))/2
                    rho_i=np.concatenate([[rho_i_1],[rho_i_2]])
                    rho=np.max(np.max(rho_i,0))
                    if rho<1:
                        if region is None:
                            region=np.array([[a,b,g,rho]])
                        else:
                            region=np.concatenate((region,
                                                   np.array([[a,b,g,rho]])),axis=0)
        return region

    def quad_agd_stab_reg(self, eigs, nbins=50):
        """
        Calculates the stable region of AGD given in [Can and Gurbuzbalaban, 2022] for convex quadratic
        objectives using grid search.
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param nbins: int
            The number of bins that is going to be used in grid search on all alpha, beta, and gamma.
        :return:
            region: np.array (alpha,beta,beta,rho)
                The list that contains the parameters alpha, beta, gamma belonging to stable set and the rate
                rho suggested by Lemma 3.1
        """
        beta = np.linspace(0,1,nbins)
        alpha= np.linspace(0,2/(min(eigs)+max(eigs)),nbins)
        # Alpha should be positive
        alpha= np.delete(alpha, np.where(alpha==0))
        # check the type of eigs:
        if type(eigs) != numpy.ndarray:
            eigs=np.array(eigs)
        region=None
        for a in alpha:
            for b in beta:
                g = b
                c_i = (1 + b) - a * (1 + g)*eigs
                d_i = -b+a*g*eigs
                rho_i_1=abs(c_i-np.emath.sqrt(c_i**2+4*d_i))/2
                rho_i_2=abs(c_i+np.emath.sqrt(c_i**2+4*d_i))/2
                rho_i=np.concatenate([[rho_i_1],[rho_i_2]])
                rho=np.max(np.max(rho_i,0))
                if rho<1:
                    if region is None:
                        region=np.array([[a,b,g,rho]])
                    else:
                        region=np.concatenate((region,
                                               np.array([[a,b,g,rho]])),axis=0)
        return region

    def quad_gd_stab_reg(self, eigs, nbins=50):
        """
        Calculates the stable region of GD given in [Can and Gurbuzbalaban, 2022] for convex quadratic
        objectives using grid search.
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param nbins: int
            The number of bins that is going to be used in grid search on all alpha, beta, and gamma.
        :return:
            region: np.array (alpha, 0, 0,rho)
                The list that contains the parameters alpha, beta, gamma belonging to stable set and the rate
                rho suggested by Lemma 3.1
        """
        alpha= np.linspace(0,2/(min(eigs)+max(eigs)),nbins)
        # Alpha should be positive
        alpha= np.delete(alpha, np.where(alpha==0))
        # check the type of eigs:
        if type(eigs) != numpy.ndarray:
            eigs=np.array(eigs)
        region=None
        for a in alpha:
            b=0
            g=b
            c_i = (1 + b) - a * (1 + g)*eigs
            d_i = -b+a*g*eigs
            rho_i_1=abs(c_i-np.emath.sqrt(c_i**2+4*d_i))/2
            rho_i_2=abs(c_i+np.emath.sqrt(c_i**2+4*d_i))/2
            rho_i=np.concatenate([[rho_i_1],[rho_i_2]])
            rho=np.max(np.max(rho_i,0))
            if rho<1:
                if region is None:
                    region=np.array([[a,b,g,rho]])
                else:
                    region=np.concatenate((region,
                                           np.array([[a,b,g,rho]])),axis=0)
        return region

    def quad_stab_region_borders(self,eigs, nbins=50):
        """
        Calculates the borders/frontier of the stable region of GMM based on quad_stab_reg().
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param nbins: float
            The number of bins that is going to be used in grid search on all alpha, beta, and gamma.
        :return:
            border: np.array (alpha, beta/gamma)
                The region alpha versus the maximum beta/gamma belonging to stable set at given alpha.
        """
        stable_region=self.quad_stab_reg(eigs,nbins)
        non_zero_gamma = np.where(stable_region[:, 2] != 0)
        ratio = np.array(stable_region[non_zero_gamma, 1] / stable_region[non_zero_gamma, 2])
        # Round the alpha to obtain the borders ofg the set
        alpha = np.round(np.array(stable_region[non_zero_gamma, 0]), 3)
        # Retrieve the borders
        border = None
        for a in np.unique(alpha):
            r = ratio[np.where(alpha == a)]
            if border is None:
                border = np.array([[a, r.max()]])
            else:
                border = np.concatenate((border, np.array([[a, r.max()]])))
        return border

    def quad_feas_region(self, stable_region, eigs, theta=1, std=1):
        """
        Calculates the feasible region given in [Can and Gurbuzbalaban, 2022. Prop 3.3].
        :param stable_region: np.array
            The stable region of GMM
        :param eigs: np.array
            The eigenvalues of the Hessian of quadratic objective.
        :param theta:
            The risk averseness parameter.
        :param std:
            The standard deviation on the noise in the gradient.
        :return:
            feas_region: np.array (alpha,beta,gamma, risk_meas, rate)
                The list of parameters alpha, beta, and gamma together with suggested risk measure and the rate.
        """
        # Check the format of stable region
        feas_region= None
        if len(stable_region.shape)==1:
            alpha, beta, gamma = stable_region
        else:
            for params in stable_region:
                # Stable region has alpha,beta,gamma, and rho.
                alpha,beta,gamma, rate = params
                risk_meas=self.quad_risk_measure(alpha, beta, gamma, eigs, theta, std)
                if risk_meas != None:
                    if feas_region is None:
                        feas_region = np.array([[alpha,beta,gamma,risk_meas,rate]])
                    else:
                        feas_region=np.concatenate((feas_region,np.array([[alpha,beta,gamma,risk_meas,rate]])))
        return feas_region

    def quad_feas_region_borders(self, e_vals, nbins=50, theta=1, std=1):
        """
        Calculate the border of the feasible region computed for smooth convex quadratic objectives.

        :param e_vals: np.array
            The eigenvalues of the Hessian of quadratic objective.
        :param nbins: float
            The number of bins for grid searching the parameters.
        :param theta: float
            The risk averseness paramters
        :param std:
            The standard deviation of noise on the gradient.
        :return:
            border: np.array (alpha, beta/gamma)
                The border of the feasible region. That is alpha versus maximum feasible beta/gamma for given alpha.
        """
        stable_region=self.quad_stab_reg(e_vals,nbins)
        feas_region=self.quad_feas_region(stable_region, e_vals, theta, std)
        # feas_region = [alpha,beta, gamma, risk measure, rate]
        non_zero_gamma= np.where(feas_region[:,2]!=0)
        ratio= np.array(feas_region[non_zero_gamma,1]/feas_region[non_zero_gamma,2])
        # Round the alpha to obtain the borders ofg the set
        alpha= np.round(np.array(feas_region[non_zero_gamma,0]),3)
        rate= np.array(feas_region[non_zero_gamma,3])
        # Retrieve the borders
        border=None
        for a in np.unique(alpha):
            r=ratio[np.where(alpha==a)]
            if border is None:
                border= np.array([[a,r.max()]])
            else:
                border= np.concatenate((border, np.array([[a,r.max()]])))

        # ratio= feas_region[:,1]/feas_region[:,2]
        # Save figure as html
        # fig.write_html(fig_path+"/str_cnvx_stable_region.html")
        # Save figure as png
        # fig.write_image(fig_path + "/quad_feasible_region.png")
        return border

    def quad_risk_measure(self, alpha, beta, gamma, eigs, theta=1, std=1):
        """
        Computes the entropic risk measure of strongly convex quadratic objectives,

        :param alpha, beta,gamma : float
            The parameters of the TMM algorithm on quadratic objective
        :param eigs: numpy.darray
            The eigenvalues of the Hessian of quadratic objective.
        :param theta:
            The risk parameter for entropic risk measure
        :param std:
            The variance bound on the noise.
        :return:

        risk_meas:   float or None
        ---------
             Returns none if parameters are not feasible, otherwise computes the entropic risk measure at given risk parameter and noise variance.
        """
        if type(eigs)!= numpy.ndarray:
            eigs=np.array(eigs)
        ratio_nom = alpha * (1 + beta - alpha * gamma * eigs)
        ratio_denom = 2*(1 - beta + alpha * gamma * eigs)*(2 * (1 + beta) - alpha * (1 + 2 * gamma) * eigs)
        if 0 in ratio_denom:
            return None
        else:
            ratio = ratio_nom/ratio_denom
            if max(ratio)<1/theta:
                risk_meas= -std ** 2 / theta * sum(np.log(1 - theta * ratio))
            else:
                risk_meas=None
            return risk_meas

    def quad_risk_meas_vs_rate_region_frontier(self, eigs, method="gmm", nbins=50, theta=1, std=1):
        """
        Calculates the frontier of the region quadratic risk measure vs convergence rate on convex quadratic objectives.

        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective function,
        :param method:
            The optimization algorithm, "gd", "agd", or "gmm".
        :param nbins:
            The number of bins used at grid search
        :param theta:
            The risk averseness parameter
        :param std:
            The standard deviation of the noise on the gradient.
        :return:
            frontier: np.array (rate, risk_measure)
                The rate versus the best risk measure found using grid search for that rate.
        """
        if method == "gmm":
            stable_region=self.quad_stab_reg(eigs,nbins)
        elif method=="agd":
            stable_region = self.quad_agd_stab_reg(eigs, nbins)
        elif method=="gd":
            stable_region = self.quad_gd_stab_reg(eigs,nbins)
        rate_sorted_index=np.argsort(stable_region[:,3])
        sorted_stable_region=stable_region[rate_sorted_index,:]
        rate_vs_risk_meas = None
        for params in sorted_stable_region:
            a,b,g,rate = params
            risk_meas=self.quad_risk_measure(a, b, g, eigs, theta, std)
            # Round the risk measure to have a better border
            if risk_meas is not None:
                # Round the rates to obtain a better border for the region
                rate=np.round(rate,2)
                if rate_vs_risk_meas is None:
                    rate_vs_risk_meas=np.array([[rate,risk_meas]])
                else:
                    rate_vs_risk_meas=np.concatenate((rate_vs_risk_meas,np.array([[rate,risk_meas]])))
        # Retrieve the frontier of the set rate_vs_risk_meas
        frontier= None
        for rho in np.unique(rate_vs_risk_meas[:,0]):
            rho_ind = np.where(rate_vs_risk_meas[:, 0] == rho)
            frontier_risk_meas = np.min(rate_vs_risk_meas[rho_ind,1])
            if frontier is None:
                frontier=np.array([[rho, frontier_risk_meas]])
            else:
                frontier=np.concatenate((frontier, np.array([[rho,frontier_risk_meas]])))
        return frontier

    def quad_evar(self, alpha, beta, gamma, eigs, nbins_theta=50, std=1, conf_level=0.95):
        """
        Computes the EVAR for convex quadratic objective by using grid search over theta to solve the minimization problem:
        ..math:
            EV@R_{1-\zeta}= \inf_{0<\theta} r_{\sigma^2}(\theta)+ \frac{2\sigma^2}{\theta}\log(1/\zeta),
        where $\zeta$ is the confidence level.

        :param alpha: float
            Learning rate of the algorithm
        :param beta: float
            Momentum parameter of the iterates of the algorithm
        :param gamma: float
            Momentum parameter of the gradient of the algorithm
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param nbins_theta:
            The number of bins for the grid searching over theta to compute EVAR by solving the minimization problem.
        :param std:
            The standard deviation of the additive Gaussian noise on the gradient of the objective.
        :param conf_level:
            The confidence level of the EV@R of the algorihtm.
        :return:
            evar: np.array
                The evar computed by solving the minimization problem using grid search over theta.
        """
        theta_range=np.linspace(0,5,nbins_theta)
        # delete theta=0
        theta_range=np.delete(theta_range,0)
        evar=float("inf")
        for theta in theta_range:
            evar_temp=self.quad_risk_measure(alpha, beta, gamma, eigs, theta, std)
            if evar_temp is not None:
                evar_temp+= 2 * std / theta * np.log(1 / conf_level)
                if evar_temp < evar:
                    evar=evar_temp

        return evar

    def frontier_quad_evar_vs_rate_region(self, eigs, method="gmm", nbins_params=50, nbins_theta=50, std=1, conf_level=0.95):
        """
        Calculates the minimum evar suggested at a given rate.

        :param eigs: np.array
            The eigenvalues of the Hessian of quadratic objective.
        :param method:
            The optimization algorithm: "gd", "agd, or "gmm".
        :param nbins_params:
            The number of bins for grid searching over the parameters alpha,beta, and gamma.
        :param nbins_theta:
            The number of bins for grid searchin over theta to find evar.
        :param std:
            The standard deviation of the additive Gaussian noise on the gradient.
        :param conf_level:
            The confidence level for the EV@R
        :return:
            frontier: np.array (rate, evar)
                The rate and the best evar computed at given rate.
        """
        if method == "gmm":
            stable_region = self.quad_stab_reg(eigs, nbins_params)
        elif method == "agd":
            stable_region = self.quad_agd_stab_reg(eigs,nbins_params)
        elif method == "gd":
            stable_region= self.quad_gd_stab_reg(eigs,nbins_params)
        rate_sorted_index = np.argsort(stable_region[:, 3])
        sorted_stable_region = stable_region[rate_sorted_index, :]
        rate_vs_evar = None

        for params in sorted_stable_region:
            a, b, g, rate = params
            evar = self.quad_evar(a, b, g, eigs, nbins_theta, std, conf_level)
            # Round the risk measure to have a better border
            if evar is not None and evar != float("inf"):
                # Round the rates to obtain a better border for the region
                rate = np.round(rate, 2)
                if rate_vs_evar is None:
                    rate_vs_evar = np.array([[rate, evar]])
                else:
                    rate_vs_evar = np.concatenate((rate_vs_evar, np.array([[rate, evar]])))

        # Retrieve the frontier of the set evar vs rate
        frontier = None
        for rho in np.unique(rate_vs_evar[:, 0]):
            rho_ind = np.where(rate_vs_evar[:, 0] == rho)
            frontier_evar = np.min(rate_vs_evar[rho_ind, 1])
            if frontier is None:
                frontier = np.array([[rho, frontier_evar]])
            else:
                frontier = np.concatenate((frontier, np.array([[rho, frontier_evar]])))
        return frontier

    def quad_evar_bound(self, alpha, beta, gamma, eigs, std, conf_level):
        """
        Calculate the evar bound for convex quadratic objectives given in [Can and Gurbuzbalaban, 2022. Theorem 1].
        :param alpha: float
            Learning rate of the algorithm
        :param beta: float
            Momentum parameter on the iterations of the algorithm.
        :param gamma: float
            Momentum parameters on the gradient of the algorithm
        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param std:
            The standard deviation of the additive Gaussian noise on the gradient of the objective.
        :param conf_level:
            The confidence level of EV@R
        :return:
            evar_bound: np.array
                The evar bound comoputed at given parameters and confidence level for noise with std provided.
        """
        if type(eigs)!= numpy.ndarray:
            eigs=np.array(eigs)
        d=len(eigs)
        ratio_denom = alpha * (1 + beta - alpha * gamma * eigs)
        ratio_nom = (1 - beta + alpha * gamma * eigs)*(2 * (1 + beta) - alpha * (1 + 2 * gamma) * eigs)
        if 0 in ratio_denom:
            return None
        else:
            u_i = ratio_nom/ratio_denom
            u_min=min(2*u_i)
            theta_0=np.log(1/conf_level)/d*(np.sqrt(1+(2*d)/np.log(1/conf_level))-1)
            evar_bound= std / (u_min * theta_0) * (-d * np.log(1 - theta_0) + 2 * np.log(1 / conf_level))
            return evar_bound

    def frontier_quad_evar_bound_vs_rate(self, eigs, method="gmm", nbins_params=50, std=1, conf_level=0.95):
        """
        Calculates frontier for the region rate vs bound on the evar of quadratic objective.

        :param eigs: np.array
            The eigenvalues of the Hessian of the quadratic objective.
        :param method:
            The optimization method: "gd", "agd", or "gmm".
        :param nbins_params:
            The number of bins to grid search over the parameters alpha, beta, and gamma.
        :param std:
            The standard deviation of the additive Gaussian noise on the gradient.
        :param conf_level:
            The confidence level of EV@R
        :return:
            frontier: np.array (rho, evar_bound)
                The list of rate and the evar bound computed at parameters giving the same rate.
        """
        if method == "gmm":
            stable_region = self.quad_stab_reg(eigs, nbins_params)
        elif method == "agd":
            stable_region = self.quad_agd_stab_reg(eigs, nbins_params)
        elif method == "gd":
            stable_region = self.quad_gd_stab_reg(eigs, nbins_params)
        rate_sorted_index = np.argsort(stable_region[:, 3])
        sorted_stable_region = stable_region[rate_sorted_index, :]
        rate_vs_evar_bound = None

        for params in sorted_stable_region:
            a, b, g, rate = params
            evar_bound = self.quad_evar_bound(a, b, g, eigs, std, conf_level)
            # Round the risk measure to have a better border
            if evar_bound is not None and evar_bound != float("inf"):
                # Round the rates to obtain a better border for the region
                rate = np.round(rate, 2)
                if rate_vs_evar_bound is None:
                    rate_vs_evar_bound = np.array([[rate, evar_bound]])
                else:
                    rate_vs_evar_bound = np.concatenate((rate_vs_evar_bound, np.array([[rate, evar_bound]])))

        # Retrieve the frontier of the set evar vs rate
        frontier = None
        for rho in np.unique(rate_vs_evar_bound[:, 0]):
            rho_ind = np.where(rate_vs_evar_bound[:, 0] == rho)
            frontier_evar = np.min(rate_vs_evar_bound[rho_ind, 1])
            if frontier is None:
                frontier = np.array([[rho, frontier_evar]])
            else:
                frontier = np.concatenate((frontier, np.array([[rho, frontier_evar]])))
        return frontier

    # Strongly convex smooth objectives
    def str_cnvx_stab_reg_params(self, L, mu, vart_nbins=100, psi_nbins=100):
        """
        Finds stable region, S_q, of GMM on strongly convex smooth non-quadratic objective
        which is given in [Can and Gurbuzbalaban, 2022. Theorem 2].
        :param L: float
            The smoothness parameter of the objective
        :param mu: float
            The strong convexity constant of the objective.
        :param vart_nbins:
            The number of bins to be used to grid search over vartheta.
        :param psi_nbins:
            The number of bins to be used to grid search over psi.

        :return:
            stable_set: np.array (vartheta, psi, rate)
                The stable region found by grid search over vartheta and psi.
        """
        kappa = L/mu
        psi_vals_low = np.linspace(0, 1, psi_nbins + 1)
        psi_vals_high = np.linspace(1, 2*kappa*(1+np.sqrt(1-1/kappa)), psi_nbins + 1)
        # Erase the 1 from arrays
        psi_vals_low = np.delete(psi_vals_low, psi_nbins)
        psi_vals_high = np.delete(psi_vals_high, 0)
        psi_vals = np.concatenate([psi_vals_low, psi_vals_high])

        # Collect stable params
        stable_vart, stable_psi, stable_rate= [], [], []

        for psi in psi_vals:
            # Set S1 as given in the paper
            if psi<1:
                if psi==0:
                    vart_vals_low = 1 / (1 + kappa * (1 - psi))
                else:
                    vart_vals_low = max(2 - 1 / psi, 1 / (1 + kappa * (1 - psi)))
                vart_vals = np.linspace(vart_vals_low, 1, vart_nbins + 1)
                # Erase 1
                vart_vals = np.delete(vart_vals, vart_nbins)
            elif psi>1:
                vart_high = min(2 - 1 / psi, 1/2*(1+np.sqrt(1+4*kappa*(psi-1))))
                vart_vals = np.linspace(1,vart_high, vart_nbins + 1)
                # Erase 1
                vart_vals = np.delete(vart_vals, 0)

            for vart in vart_vals:
                alpha= (1-vart)/(L*(1-psi))
                m_psi= mu*(psi**2)-L*(1-psi)**2
                rho2=1-np.sqrt(vart*alpha*mu)
                condition= rho2*(1-alpha*m_psi/vart)-(1-alpha*psi*mu)**2
                if condition <=0:
                    stable_vart = np.append(stable_vart, vart)
                    stable_psi = np.append(stable_psi, psi)
                    stable_rate = np.append(rho2,stable_rate)

        sort_ind = np.argsort(stable_psi)
        stable_set = np.concatenate([[stable_vart[sort_ind]], [stable_psi[sort_ind]],[stable_rate[sort_ind]]])
        # stable_set=([[stable_vart],[stable_psi], [stable_rate]]) where psi !=1
        return stable_set
    
    def str_cnvx_agd_stab_reg_params(self, L, mu, alpha_nbins=100):
        """
        Finds stable region, S_q, of AGD on strongly convex smooth non-quadratic objective
        which is given in [Can and Gurbuzbalaban, 2022. Theorem 2].
        :param L: float
            The smoothness parameter of the objective
        :param mu: float
            The strong convexity constant of the objective.
        :param alpha_nbins: float (default: 100)
            The number of bins to be used to grid search over vartheta.
        :return:
            stable_set: np.array (alpha, rate)
                The stable region found by grid search over alpha.
        """

        kappa = L/mu
        vart, psi=1, 1
        alpha_lin=np.linspace(0,2/(L+mu),alpha_nbins)
        # Erase 0 from alpha
        alpha_lin=np.delete(alpha_lin,0)
        # Collect stable params
        stable_alpha, stable_rate= [], []

        for alpha in alpha_lin:
            # Set S0 as given in the paper
            m_psi= mu*(psi**2)-L*(1-psi)**2
            rho2=1-np.sqrt(vart*alpha*mu)
            condition= rho2*(1-alpha*m_psi/vart)-(1-alpha*psi*mu)**2
            if condition <=0:
                stable_alpha = np.append(stable_alpha, alpha)
                stable_rate = np.append(rho2,stable_rate)

        sort_ind = np.argsort(stable_alpha)
        stable_set = np.concatenate([[stable_alpha[sort_ind]],[stable_rate[sort_ind]]])
        # stable_set=([[stable_alpha],[stable_rate]])
        return stable_set
    
    
    def str_convx_MI_check(self, L, mu, vart, psi):
        """
        This function checks the MI inequality for alpha, beta, and gamma defined as
        in [Can and Gurbuzbalaban, 2022. (MI)] with respect to vartheta and psi.
        :param L: float
            The smoothness constant of the objective.
        :param mu: float
            The strong convexity constant of the objective.
        :param vart: float
            The parameter of the algorithm
        :param psi: float
            The parameter of the algorithm.
        :return:
            V: np.array
                The matrix M-X, suggested by the MI which should be negative semi-definite.
            eigs: np.array
                The eigenvalues of the matrix (sorted).
        """
        kappa = L/mu
        alpha = (1 - vart)/(L * (1 - psi))
        p = np.sqrt(vart/(2 * alpha))
        p0 = np.sqrt(mu/2)

        rho2 = 1 - np.sqrt(alpha*vart*mu)
        beta = rho2 / (1 - alpha * psi * mu) * (1 - np.sqrt(alpha * mu / vart))
        gamma = psi*beta

        P = np.array([[p], [-p + p0]])
        P= np.matmul(P,P.T)
        A=np.array([[1 + beta, -beta],[1, 0]])
        B = np.array([[-alpha],[0]])
        C = np.array([[(1 + gamma)],[-gamma]])
        delta = beta - gamma
        M1 = np.matmul(A.T,np.matmul(P,A))-rho2*P
        M2 = np.matmul(B.T,np.matmul(P,A))
        M3 = np.matmul(B.T,np.matmul(P,B))
        M1shape= M1.shape
        M2shape= M2.shape
        M3shape= M3.shape
        M =np.zeros((M1shape[0]+M2shape[0], M1shape[1]+M2shape[0]))
        M[0:M1shape[0],0:M1shape[1]]=M1
        M[0:M1shape[0],M1shape[1]:(M1shape[1]+M2shape[0])]=M2.T
        M[M1shape[0]:(M1shape[0]+M2shape[0]),0:M2shape[1]]=M2
        M[M1shape[0]:(M1shape[0]+M3shape[0]),M1shape[1]:(M1shape[1]+M3shape[1])]=M3

        X1 = 1/2 * np.array([[-L*delta**2,L*delta**2,-(1 - alpha * L) * delta],
                            [L*delta**2, -L*delta**2, (1 - alpha * L) * delta],
                            [-(1-alpha*L)*delta, (1-alpha*L)*delta, alpha*(2-alpha*L)]])

        X2 = 1 / 2 * np.array([[gamma**2 * mu, -gamma**2 * mu, -gamma],
                               [-gamma**2*mu, gamma**2*mu, gamma],
                               [-gamma, gamma, 0]])

        X3 = 1 / 2 * np.array([[(1 + gamma)**2*mu, -gamma*(1+gamma)*mu, -(1+gamma)],
                                [-gamma * (1 + gamma) * mu, gamma**2*mu, gamma],
                                [-(1 + gamma), gamma, 0]])
        X=X1+rho2*X2+(1-rho2)*X3
        V=M-X
        return V, np.linalg.eig(V)[0]

    def str_cnvx_evar_bound(self, L, mu, varth, psi, dimension, conf_lev=0.99, std=1):
        """
        Calculates the evar bound provided for strongly convex smooth non-quadratic objectives
        in [Can and Gurbuzbalaban, 2022. Theorem 3]
        :param L: float
            The smoothness parameter of the objective function.
        :param mu: float
            The strong convexity constant of the objective.
        :param varth: float
            The parameter of the algorithm.
        :param psi: float
            The parameter of the algorithm.
        :param dimension: int
            The dimension of the problem, i.e. the number of features.
        :param conf_lev: float
            The confidence level of EV@R
        :param std: float
            The standard deviation of the additive Gaussian noise on the gradient.
        :return:
            evar_bound: np.array
                The evar bound computed at given parameters.
        """
        d=dimension
        kappa=L/mu
        varphi=0.99
        if psi!=1:
            alpha=(1-varth)/(L*(1-psi))
            beta=1-np.sqrt(varth*alpha*mu)/(1-alpha*psi*mu)*(1-np.sqrt((alpha*mu)/varth))
            gamma=psi*beta

        # Define v_{\vartheta,\psi}
        v_vp=2*(L**2)/mu*(2*(beta-gamma)**2+(1-alpha*L)**2*(1+2*gamma+2*(gamma**2)))\
            +0.5*varth/alpha*(1-np.sqrt(varth*alpha*mu))

        # Define the Theta_u^{g} and \theta_\varphi^{g} as defined in equation (4.7)
        theta_u = np.sqrt(varth * mu)/(alpha*(8*v_vp*np.sqrt(alpha)+alpha*np.sqrt(varth*mu)*(varth+alpha*L)))
        theta_var=varphi*theta_u

        bbrho_1 = 0.5 * (1 - np.sqrt(varth * alpha*mu)+(theta_var*4*alpha**2*v_vp)/(2-theta_var*alpha*(varth+alpha*L)))
        bbrho_2=bbrho_1**2+(16*theta_var*alpha**2*v_vp)/(2-theta_var*alpha*(varth+alpha*L))
        bbrho=0.5*bbrho_1+0.5*np.sqrt(bbrho_2)

        if np.log(1/conf_lev)< 0.5*d/(1-bbrho)*((theta_var*alpha*(varth+alpha*L))/(2-theta_var*(varth+alpha*L)))**2:
            out=0.5*alpha*(varth+alpha*L)*(np.sqrt(d/(1-bbrho))+np.sqrt(2*np.log(1/conf_lev)))**2
        else:
            out=d*alpha*(varth+alpha*L)/((1-bbrho)*(2-theta_var*alpha*(varth+alpha*L)))\
                +2*np.log(1/conf_lev)/theta_var
        return std * out

    def str_cnvx_agd_evar_bound(self, L, mu, alpha, dimension, conf_lev=0.95, std=1):
        """
        Calculates the evar bound provided for AGD on strongly convex smooth non-quadratic objectives
        in [Can and Gurbuzbalaban, 2022. Theorem 3]
        
        :param L: float
            The smoothness parameter of the objective function.
        :param mu: float
            The strong convexity constant of the objective.
        :param alpha: float
            The parameter of the algorithm.
        :param dimension: int
            The dimension of the problem, i.e. the number of features.
        :param conf_lev: float
            The confidence level of EV@R
        :param std: float
            The standard deviation of the additive Gaussian noise on the gradient.
        :return:
            evar_bound: np.array
                The evar bound computed at given parameters.
        
        """
        d = dimension
        kappa = L / mu
        psi,varth=1, 1
        if np.log(1 / conf_lev) < d / (2 * mu * alpha * varth):
            out = np.sqrt(varth * alpha) * (np.sqrt(2 * np.log(1 / conf_lev)) + np.sqrt(d)) ** 2
        else:
            out = (1 + np.sqrt(alpha * varth * mu)) * \
                  (d / np.sqrt(alpha * varth * mu) + 2 * np.log(1 / conf_lev))
        return 0.5 * (std**2) * out / np.sqrt(mu)
    
    
    # Boundary smoothing to compensate for the grid-search
    def smoothTriangle(self, data, degree):
        """
        This part is to smooth out the region data that has been computed using grid search.
        :param data: np.array
            The region data
        :param degree: int
            The degree of smoothing, i.e. averaging scale.
        :return:
            smoothed: np.array
                The smoothed region.
        """
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))  # up then down
        smoothed = [data[0]]
        for i in range(degree + 1, len(data) - degree * 2):
            point = data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point) / np.sum(triangle))
        # Handle boundaries
        smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed

    def str_cnv_stable_region_frontier(self, L, mu,vartbins=250, psibins=250):
        """
        Finds the stable region suggested by Theorem 2.

        :param L: float
             The smoothness parameter of the strongly convex smooth non-quadratic objective.
        :param mu:
            The stong convexity paramter of the objective.
        :param vartbins:
            The number of bins to be used to grid search over vartheta.
        :param psibins:
            The number of binst to be used to grid search over psi.
        :return:
            region: np/array (alpha, min(beta/gamma), max(beta/gamma))
                The frontier of the stable region with respect to alpha and the ratio beta/gamma which are defined as
                 a function of vartheta and psi as given in [Can and Gurbuzbalaban, 2022. Theorem 2].
        """
        region = self.str_cnvx_stab_reg_params(L, mu, vart_nbins=vartbins, psi_nbins=psibins)
        # region = (vartheta; psi)
        psi=region[1]
        alpha = (1 - region[0]) / (L * (1 - region[1]))
        alpha = np.round(np.array(alpha),2)
        rate = 1 - np.sqrt(region[0] * alpha * mu)
        rate_opt = 1 - np.sqrt(mu / L)
        rate /= rate_opt
        border=None
        for a in np.unique(alpha):
            # Retrieve the psi values for same a
            ind_a=np.where(alpha == a)
            psi_a= psi[ind_a]
            if border is None:
                border= np.array([[a, psi_a.min(), psi_a.max()]])
            else:
                border=np.concatenate((border, np.array([[a,psi_a.min(),psi_a.max()]])))
        # Smooth out
        for ind, bounds in enumerate(border):
            if ind<len(border)-2 and ind>0:
                if border[ind][2]<border[ind+1][2]:
                    border[ind][2]=1/4*(border[ind-2][2]+border[ind-1][2]+border[ind+1][2]+border[ind+2][2])
        # Neighborhood averaging
        for ind, bounds in enumerate(border):
            if ind<len(border)-3 and ind>1:
                border[ind][2]=1/4*(border[ind-2][2]+border[ind-1][2]+border[ind+1][2]+border[ind+2][2])
        for ind, bounds in enumerate(border):
            if ind<len(border)-2 and ind>0:
                border[ind][2]=1/3*(border[ind-1][2]+border[ind][2]+border[ind+1][2])

        # Add more alpha into region
        new_border= None
        for ind, bounds in enumerate(border):
            if new_border is None:
                new_border=bounds.reshape(1,len(bounds))
            else:
                new_border=np.concatenate((new_border,bounds.reshape(1,len(bounds))))
            if ind>0 and ind<len(border)-2:
                a=0.5*(border[ind][0]+border[ind+1][0])
                psi_min= 0.5*(border[ind][1]+border[ind+1][1])
                psi_max=0.5*(border[ind][2] + border[ind+1][2])
                new_border=np.concatenate((new_border,np.array([[a,psi_min,psi_max]])))

        # Write over the border parameter
        border=new_border
        end=len(border[:,2])-55
        # Triangular smoothing out
        border[53:end,2]=self.smoothTriangle(border[53:end,2],10)
        # border[:,2]=self.smoothTriangle(border[:,2],5)
        return border

    def rate_vs_evar_bound_frontier(self, L, mu, dimension, std=1, conf_level=0.95, varthbins=250):
        """
        Finds the rate and evar bound for agd at given parameters so that the comparison rate versus evar bound can be done.
        :param L: float
            The smoothness parameter of the objective.
        :param mu: float
            The strong convexity parameter of the objective.
        :param dimension: int
            The dimension of the problem, i.e. the number of features.
        :param std: float
            The stadnard deviation of the additive Gaussian noise on the gradient.
        :param conf_level: float
            The confidence level of the EV@R.
        :param varthbins:
            The number of bins to grid search over vartheta.
        :return:
            frontier: np.array (alpha,rate,evar)
                The list of alpha, rate and evar suggested by given alpha.
        """
        psi=1
        varth= np.linspace(0,1/L,varthbins)
        varth= np.delete(varth,len(varth)-1)
        varth= np.delete(varth,0)
        frontier= None
        for v in varth:
            evar= self.str_cnvx_evar_bound(L, mu, v, psi, dimension, conf_level, std)
            a = (1 - v) / L
            rate= 1-np.sqrt(a*v*mu)
            if frontier is None:
                frontier=np.array([[a,rate,evar]])
            else:
                frontier=np.concatenate((frontier, np.array([[a,rate,evar]])))
        return frontier
