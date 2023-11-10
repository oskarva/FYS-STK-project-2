import autograd.numpy as np

def CostOLS(target):
    """
    Description:
    ------------
        Makes a Cost function for OLS with the target vector baked in for ease of derivating with respect to X.

    Parameters:
    ------------
        I   target (np.ndarray): The target data, matrix with shape n x m

    Returns:
    ------------
        I   func (Callable): Cost function. Takes in X and returns a float.
    """
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):
    """
    Description:
    ------------
        Makes a Cost function for Logistic Regression with the target vector baked in for ease of derivating with respect to X.

    Parameters:
    ------------
        I   target (np.ndarray): The target data, matrix with shape n x m

    Returns:
    ------------
        I   func (Callable): Cost function. Takes in X and returns a float.
    """
    def func(X):
        
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func


def CostCrossEntropy(target):
    """
    Description:
    ------------
        Makes a Cost function for Cross Entropy with the target vector baked in for ease of derivating with respect to X.

    Parameters:
    ------------
        I   target (np.ndarray): The target data, matrix with shape n x m

    Returns:
    ------------
        I   func (Callable): Cost function. Takes in X and returns a float.
    """
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func