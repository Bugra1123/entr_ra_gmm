# Install packages
import numpy as np
import numpy.linalg as nplin
import numpy.random as nprand

# Quadratic cost function
def quadratic_cost(x:float,eigs=None, regulizer=1,sigma=float(1), seed=None):
    """
    The quadratic function to be used at the optimization problem.

    :param x:      [np.array]   (# of params) The parameters of the function
    :param eigs:   [np.array]   (# of eigenvalues) The eigenvalues of the Hessian of the quadratic objective
    :param sigma:  [float]       The variance of the additive Gaussian unbiased noise on the gradient.
    :return:
    y:             [float]       The function value computed at x.
    grad_y:        [np.array]   The noisy gradient computed at x.
    """
    if seed is not None: nprand.seed(seed)
    num_feat: int = len(x)
    if eigs is None: eigs = np.ones(shape=num_feat)
    # if x_optim is None: x_optim = np.ones(shape=num_feat)
    Q = np.diag(eigs)
    y = np.matmul(Q,x)
    b=np.ones((num_feat,1))
    b/=np.linalg.norm(b)
    grad_y = y + b + regulizer*x + np.random.normal(0, sigma, size=(len(x),1))
    y = 0.5 * np.matmul(x.T, y) + np.matmul(x.T,b)+ 0.5*regulizer*np.matmul(x.T,x)
    return y[0][0], grad_y

# The sigmoid function.
def sigmoid(w, Data, Label):
    # TODO: Generate a derivation file.
    out=1/(1 + np.exp(-Label*(np.matmul(Data, w))))
    return out

# Logistic regression cost
def logistic_cost(w, Data, Outcome, reg=float(1), sigma=float(1), seed=None):
    """
    The logistic regression function:
        .. math::
            \frac{1}{N}\sum_{i}^{N}\log(1+\exp\{-y_i(x_i^\top w)\})+0.5reg ||w||^2
    and its gradient.
    # TODO Add reference for the function
    :param w:       np.array (# of features, 1)
        The variable of the function.
    :param Data:    np.array (#of samples, # of feature)
        The data consists of samples.
    :param Outcome: np.array (# of samples)
        The binary outcome data.
    :param reg:     float (default: 1)     
        The regularization factor for the logistic regression.
    :param sigma:   float (default: 1 )     
        The variance on the unbiased Gaussian gradient noise.
    :param seed:    int ( default: None)
        Random seed.
    :return:
    func_value:     float 
        The value of the function computed at w.
    grad_vec  :     np.array (# of features, 1) 
        The noisy gradient computed at w.

    """
    if seed is not None: nprand.seed(seed)
    # Retrieve the data information
    s,f=Data.shape
    # Compute the logistic function at each sample
    cost= -np.log(sigmoid(w, Data, Outcome))
    # Compute the average total cost
    func_value=cost.mean()+reg*0.5*nplin.norm(w)**2
    # Compute the gradient
    grad_vec= 1/s*np.matmul(-np.transpose(Data), Outcome * (1 - sigmoid(w, Data, Outcome)))\
          + reg * w + nprand.normal(0, sigma, size=(f, 1))
    return func_value, grad_vec
