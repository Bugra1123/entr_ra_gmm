# Install packages
import numpy as np
import numpy.linalg as nplin
import numpy.random as nprand

def Create_Regression_Data(Variance, Sample, Num_Feature, seed=None):
    """
    Generate synthetic regression data from a normal distribution.

    :param Variance: float
        The variance of the generating normal distribution.
    :param Sample:  int
        The number of samples.
    :param Num_Feature:
        The number of features each data has.
    :param seed: int (default: None)
        The random generator seed
    :return:
    X:  np.array (# of samples, # of features)
        The synthetic data.
    Y:  np.array (# of samples, (1 or -1))
        The output/dependent variables
    x_opt: np.array (# of feature, 1)
        The optimal solution.
    """
    if seed is not None: nprand.seed(seed)
    # Create an artifical feature vector.
    x_opt=np.random.normal(1,Variance,size=(Num_Feature,1))
    x_opt=x_opt/nplin.norm(x_opt)
    # Generate the sample data.
    X=nprand.normal(1,Variance,size=(Sample,Num_Feature))
    # Normalize sample data.
    X=1/abs(X).max()*X
    # Generate outcomes
    Y=np.matmul(X,x_opt)
    return X, Y, x_opt

def Create_Classification_Data(Std, Sample, Num_Feature, seed=None):
    """
    Generate synthetic classification data from a normal distribution which is labeled +1 or -1.

    :param Std: float
        The standard deviation of the generating normal distribution.
    :param Sample:  int
        The number of samples.
    :param Num_Feature:
        The number of features each data has.
    :param seed: int (default: None)
        The random generator seed
    :return:
        X:  np.array (# of samples, # of features)
            The synthetic data.
        Y:  np.array (# of samples, (1 or -1))
            The labels randomly assigned to each datapoint.
        x_opt: np.array (# of feature, 1)
            The artificial parameter vector sampled from normal distribution with mean 1 and Variance, to generate synthetic
            data
    """
    if seed is not None: nprand.seed(seed)
    # Create an artifical parameter vector.
    x_0=np.random.normal(0, Std, size=(Num_Feature, 1)).astype("float32")
    # Generate the sample data.
    X=np.random.normal(0, Std, size=(Sample, Num_Feature)).astype("float32")
    # Generate outcomes
    Out=np.matmul(X,x_0)
    # Get rid of any zeros
    Out+=0.001*abs(Out).min()
    Y=np.sign(Out)
    return X, Y, x_0
