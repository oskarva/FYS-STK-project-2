import numpy as np


def ols_regression(X_train, X_test, z_train, z_train_mean=0):
    '''
    This code performs Ordinary Least Squares regression
    It takes the design matrices for the training and test sets, the training set response, and a flag indicating whether the data is scaled as input
    It returns the approximation of the response for the training and test sets, as well as the estimated beta values
    
    Parameters:
        X_train (two dimensional numpy matrix): Design matrix with training set
        X_test (two dimensional numpy matrix): Design matrix with test set
        z_train (numpy array): Response for training set
        z_train_mean (float): Mean of the response for training set. Default is 0.
        scaled (bool): Flag indicating whether the data is scaled. Default is False.

    Returns:
        ztilde_train_OLS (numpy array): approximation of response for training set
        ztilde_test_OLS (numpy array): approximation of response for test set
        betas (numpy array): estimated beta values
    '''
    betas = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ z_train # estimated beta values
    ztilde_train_OLS = (X_train @ betas) + z_train_mean # approximation of response using the design matrix and the estimated beta values
    ztilde_test_OLS = (X_test @ betas) + z_train_mean # prediction of response using the design matrix and the estimated beta values
    return ztilde_train_OLS, ztilde_test_OLS, betas











