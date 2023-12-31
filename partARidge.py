"""
This code tests different gradient descent methods on the same data to compare convergance rates.
We optimize for the Ridge cost function on a dataset of 100 samples of a 
second degree polynomial.

For the stochastic part we use 10 batches. 
"""


from cost_functions import *
from schedulers import *
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def cost_OLS(y, X, theta):
    '''
    Description:
    ------------
    Cost function for OLS regression.

    Parameters:
    ------------
        I   y (np.ndarray): The target vector, with length n
        II  X (np.ndarray): The design matrix, with n rows of p features each
        III theta (np.ndarray): The regressin parameters, with length p

    Returns:
    ------------
        I   z (float): Cost, is the MSE between predicted X*theta and target y
    '''
    return 1/y.size * np.sum((y - X @ theta)**2)


def grad_cost_Ridge_lam(lamb):
    '''
    Description:
    ------------
    Makes a function with gradient in terms of y, X, theta for Ridge regression.

    Parameters:
    ------------
        I   y (np.ndarray): The target vector, with length n
        II  X (np.ndarray): The design matrix, with n rows of p features each
        III theta (np.ndarray): The regressin parameters, with length p

    Returns:
    ------------
        I   grad_cost_Ridge (Callable): Gradient function.
    '''
    def grad_cost_Ridge(y, X, theta):
        return 2/y.size * X.T @ ( (X @ theta) -y) + lamb * 2 * theta
    
    return grad_cost_Ridge


#Plotting
import os
from pathlib import Path
import seaborn as sns

sns.set_theme()

#Define path for saving plots
cwd = os.getcwd()
path = Path(cwd) / "FigurePlots" / "Gradienet_descent_MSE_Ridge"
if not path.exists():
    path.mkdir()

def plot_etas_lams(etas, lams, MSEs, title, filename):
    '''
    Description:
    ------------
        Function for plotting MSE by eta values and lambda values.
    Parameters:
    ------------
        I   etas list(floats): eta values tested
        II  lams list(floats): lambda values tested
        III  MSEs list(floats): MSE for the eta values
        IV title (str): Title for plot
        V filename (str) | Path: path for saving figure

    Returns:
    ------------
        I   None
    '''
    plt.figure()
    ax = sns.heatmap(MSEs, xticklabels=etas, yticklabels=lams, annot=True, robust = True)
    ax.set(xlabel="Lambda", ylabel="Etas")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path / filename)
    plt.close()

def fit(X, y, grad_cost_func, scheduler, batches = 1, epochs = 100, seed = 13):
    '''
    Description:
    ------------
        Function for fitting regression parameters. Does a gradient descent with choosen method.

    Parameters:
    ------------
        I   X (np.ndarray): The design matrix, with n rows of p features each
        II  y (np.ndarray): The target vector, with length n
        III grad_cost_func (Callable): gradient of cost function.

    Optional Parameters:
    ------------
        IV batches  (int): Number of minibatches to split data into. If 1, just normal gradient descent
        V epochs (int): Number of iterations to do.
        VI seed (int): Seed to inititalze random numbers.

    Returns:
    ------------
        I   theta (np.ndarray): Optimized theta paramaters.
    '''

    np.random.seed(seed)
    batch_size = int(np.shape(X)[0]/batches)
    theta = np.random.rand(X.shape[1], 1)

    for i in range(epochs):
        for i in range(batches):
             # allows for minibatch gradient descent loops over all batches
            if i == batches - 1:
                 # If the for loop has reached the last batch, take all thats left
                X_batch = X[i * batch_size :, :]
                y_batch = y[i * batch_size :]
            else:
                X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                y_batch = y[i * batch_size : (i + 1) * batch_size]
            gradient = grad_cost_func(y_batch, X_batch, theta)
            change = scheduler.update_change(gradient)
            theta  = theta - change
            scheduler.reset
    
    return theta
            


def f(x):
    '''
    Description:
    ------------
    Second degree polynomial

    Parameters:
    ------------
        I   x (np.ndarray): X values
    Returns:
    ------------
        I   f(x) (np.ndarray): Function values
    '''
    return -3+5 * x -3*x**2

#Keep track of MSE over all runs
MSE_best_by_method_all = []
MSE_method_names = []


#Run experiment for 5 different seeds.
seeds = range(5)
for seed in seeds:
    np.random.seed(seed)   

    n_datapoints = 100
    x = np.random.rand(n_datapoints, 1)
    y = f(x) + 0.1*np.random.randn(n_datapoints, 1)
    X = np.c_[np.ones(n_datapoints), x, x**2]

    #Define hyperparameters to be used.
    n_etas = 10
    n_lams = 9
    etas = [10**(-k) for k in range(n_etas)]
    lams = [10**(-k) for k in range(n_lams)]
    batches = 10
    epochs = 20
    n_methods = 8
    mom = 0.9

    #Keep track of best achieved MSE this run
    MSE_best_by_method = []
    MSE_method_names = []

    '''
    Next part runs though different gradient descent methods including
    Not stochastic descent:
    Constant
    Momemtum

    Stochastic Gradient Descent:
    Constant
    Momemtum
    Adagrad
    Adagrad Momemtum
    RMSprop
    Adam
    '''

    # Constant 
    method_name = "Constant"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Constant(eta)
            theta = fit(X, y, grad_cost_Ridge_lam(lam), scheduler, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Constant")
    #print(f"MSE constant: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


    #Momentum
    method_name = "Momemtum"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Momentum(eta, mom)
            theta = fit(X, y, grad_cost_Ridge_lam(lam), scheduler, epochs=epochs)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Momemtum")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


    #Stochastic constant
    method_name = "Constant_SGD"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Constant(eta)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Constant SGD")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


    #SGD with momemtum
    method_name = "Momemtum_SGD"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Momentum(eta, mom)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Momemtum SGD")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")



    # 4 Adagrad no momemtum
    method_name = "Adagrad"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Adagrad(eta)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Adagrad")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


    #Adagrad with momemtum
    method_name = "AdagradMomemtum"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = AdagradMomentum(eta, mom)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Adagrad mom")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")



    # 5 RSMProp
    method_name = "RMSprop"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = RMS_prop(eta, 0.9)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("RMSProp")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")



    # Adam
    method_name = "Adam"
    MSEs = np.zeros((n_etas, n_lams))
    for i, eta in enumerate(etas):
        for j, lam in enumerate(lams):
            scheduler = Adam(eta, 0.9, 0.99)
            theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
            #print(theta)
            MSEs[i][j] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(np.min(MSEs, axis = 0))
    MSE_method_names.append("Adam")
    #print(f"MSE momentum: {MSEs}")
    plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


    #Add results of this run to list
    MSE_best_by_method_all.append(MSE_best_by_method)



MSE_best_by_method = np.asarray(MSE_best_by_method_all)

#Average over all the different starting seeds
MSE_best_by_method = np.mean(MSE_best_by_method, axis=0)

#Print array of all the best cost function value, minimized over eta value, axis are methods and lambda values.
print(f"\n The methods tested are: \n {MSE_method_names}")
print(f"Best MSE array: \n {np.around(MSE_best_by_method, decimals = 3)}")


#Define path for plot
path = Path(cwd) / "FigurePlots"

if not path.exists():
    path.mkdir()

#Make heatmap with methods along on axis and lambda values along the other axis. The cells are the lowest Cost 
#function (MSE) over all eta values.

plt.figure()
ax = sns.heatmap(np.asarray(MSE_best_by_method),
                  yticklabels=MSE_method_names, 
                  xticklabels=lams,
                  annot=np.asarray(MSE_best_by_method))
ax.set(xlabel="Gradient descent methods", ylabel="Lambda value")
plt.title("Best MSE error by gradient descent method for Ridge")
plt.savefig(path / "PartARidgeHeatMap.png")




