"""
This code tests different gradient descent methods on the same data for a comparison.
We optimize for the OLS cost function on a dataset of 100 samples of a 
second degree polynomial.

For the stochastic part we use 10 batches. 

We repeat this experiment for 5 different starting seeds and take an average for each method.
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

def grad_cost_OLS(y, X, theta):
    '''
    Description:
    ------------
    Gradient of Cost function for OLS regression with respect to theta variable evaluted at y, X, theta.

    Parameters:
    ------------
        I   y (np.ndarray): The target vector, with length n
        II  X (np.ndarray): The design matrix, with n rows of p features each
        III theta (np.ndarray): The regressin parameters, with length p

    Returns:
    ------------
        I   grad(C) (np.ndarray): Gradient of cost function.
    '''
    return 2/y.size * X.T @ ((X @ theta) - y)


#Define path for plotting
import os
from pathlib import Path
cwd = os.getcwd()
path = Path(cwd) / "FigurePlots" / "Gradienet_descent_MSE"

if not path.exists():
    path.mkdir()

def plot_etas(etas, MSEs, title, filename):
    '''
    Description:
    ------------
    Function for plotting MSE by eta values.

    Parameters:
    ------------
        I   etas list(floats): eta values tested
        II  MSEs list(floats): MSE for the eta values
        III title (str): Title for plot
        IV filename (str | Path): path for saving figure

    Returns:
    ------------
        I   None
    '''
    plt.figure()
    plt.plot(etas, MSEs)
    plt.xscale("log")
    plt.ylim(0, 10)
    plt.title(title)
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
             # allows for minibatch gradient descent
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
            

#Function we base our data on
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

#Keep track of MSE
MSE_best_by_method_all = []
MSE_method_names = []

#Run the experiment for 5 different starting seeds.
seeds = range(5)

for seed in seeds:

    # Data generation
    np.random.seed(seed)
    n_datapoints = 100
    x = np.random.rand(n_datapoints, 1)
    y = f(x) + 0.1*np.random.randn(n_datapoints, 1)
    X = np.c_[np.ones(n_datapoints), x, x**2]

    #Define parameters for gradient descent
    n_etas = 10
    etas = np.geomspace(10**(-5), 2, n_etas)
    epochs = 20
    batches = 10


    #Keep track of best achieved MSE this run
    MSE_best_by_method = []
    MSE_method_names = []

    '''
    Next part runs though different gradient descent methods including
    Not stochastic:
    Constant
    Momemtum

    Stochastic gradient descenet:
    Constant
    Momemtum
    Adagrad
    Adagrad Momemtum
    RMSprop
    Adam
    '''


    # Constant 
    method_name = "Constant"
    MSEs = [0] * n_etas
    for i, eta in enumerate(etas):
        scheduler = Constant(eta)
        theta = fit(X, y, grad_cost_OLS, scheduler, epochs=epochs)
        MSEs[i] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {MSEs}")
    plot_etas(etas, MSEs, method_name, method_name + ".png")

    #Momentum
    method_name = "Momentum"
    n_moms = 9
    mom_params = np.linspace(0.1, 1, n_moms)
    MSEs = np.zeros((n_etas, n_moms))

    for i, eta in enumerate(etas):
        for j, mom in enumerate(mom_params):
            scheduler = Momentum(eta, mom)
            theta = fit(X, y, grad_cost_OLS, scheduler, epochs=epochs)
            MSEs[i][j] = cost_OLS(y, X, theta)


    MSE_best_by_method.append(np.min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {np.min(MSEs)}")
    plot_etas(etas, np.min(MSEs, axis=1), method_name, method_name + ".png")


    #Stochastic part

    #Stochastic constant
    method_name = "Constant_SGD"
    MSEs = [0] * n_etas
    for i, eta in enumerate(etas):
        scheduler = Constant(eta)
        theta = fit(X, y, grad_cost_OLS, scheduler,batches =batches, epochs=epochs)
        MSEs[i] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {MSEs}")
    plot_etas(etas, MSEs, method_name, method_name + ".png")


    #SGD with momentum
    method_name = "Momentum_SGD"
    n_moms = 9
    mom_params = np.linspace(0.1, 1, n_moms)
    MSEs = np.zeros((n_etas, n_moms))

    for i, eta in enumerate(etas):
        for j, mom in enumerate(mom_params):
            scheduler = Momentum(eta, mom)
            theta = fit(X, y, grad_cost_OLS, scheduler, epochs=epochs, batches=batches)
            MSEs[i][j] = cost_OLS(y, X, theta)


    MSE_best_by_method.append(np.min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {np.min(MSEs)}")
    plot_etas(etas, np.min(MSEs, axis=1), method_name, method_name + ".png")

    # 4 Adagrad no momemtum
    method_name = "Adagrad"
    MSEs = [0] * n_etas
    for i, eta in enumerate(etas):
        scheduler = Adagrad(eta)
        theta = fit(X, y, grad_cost_OLS, scheduler, batches =batches, epochs=epochs)
        MSEs[i] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {MSEs}")
    plot_etas(etas, MSEs, method_name, method_name + ".png")


    #Adagrad with momentum
    method_name = "AdagradMomentum"
    n_moms = 9
    mom_params = np.linspace(0.1, 1, n_moms)
    MSEs = np.zeros((n_etas, n_moms))

    for i, eta in enumerate(etas):
        for j, mom in enumerate(mom_params):
            scheduler = AdagradMomentum(eta, mom)
            theta = fit(X, y, grad_cost_OLS, scheduler, epochs=epochs, batches=batches)
            MSEs[i][j] = cost_OLS(y, X, theta)


    MSE_best_by_method.append(np.min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {np.min(MSEs)}")
    plot_etas(etas, np.min(MSEs, axis=1), method_name, method_name + ".png")



    # 5 RSMProp
    MSEs = [0] * n_etas
    method_name = "RMSprop"
    for i, eta in enumerate(etas):
        scheduler = RMS_prop(eta, 0.9)
        theta = fit(X, y, grad_cost_OLS, scheduler,batches =batches, epochs=epochs)
        MSEs[i] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {MSEs}")
    plot_etas(etas, MSEs, method_name, method_name + ".png")



    # Adam
    method_name = "Adam"
    MSEs = [0] * n_etas
    for i, eta in enumerate(etas):
        scheduler = Adam(eta, 0.9, 0.99)
        theta = fit(X, y, grad_cost_OLS, scheduler,batches =batches, epochs=epochs)
        MSEs[i] = cost_OLS(y, X, theta)

    MSE_best_by_method.append(min(MSEs))
    MSE_method_names.append(method_name)
    #print(f"MSE {method_name}: {MSEs}")
    plot_etas(etas, MSEs, method_name, method_name + ".png")


    #print(f"\n MSE list for seed {seed}: {MSE_best_by_method}\n")


    #Add results of this run to list
    MSE_best_by_method_all.append(MSE_best_by_method)


MSE_best_by_method = np.asanyarray(MSE_best_by_method_all)

#Average over all the runs.
MSE_best_by_method = np.mean(MSE_best_by_method, axis=0)

print(f"\n List of methods tested: \n {MSE_method_names}")
print(f"\n Average best MSE for methods:\n {np.around(MSE_best_by_method, decimals=3)}\n")

#Define path for plotting
path = Path(cwd) / "FigurePlots"
if not path.exists():
    path.mkdir()


#Plots the average best achieved MSE by gradient method
plt.figure()
plt.bar(range(0, 2*len(MSE_method_names), 2), MSE_best_by_method,  tick_label =MSE_method_names)
plt.title("Average best MSE error by gradient descent method OLS")
plt.xticks(rotation=-30)
plt.tight_layout()
plt.savefig(path / "PartAOLS.png")





