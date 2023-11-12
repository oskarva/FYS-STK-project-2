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

def grad_cost_Ridge_lam(lamb):
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
    def grad_cost_Ridge(y, X, theta):
        return 2/y.size * X.T @ ( (X @ theta) -y) + lamb * 2 * theta
    
    return grad_cost_Ridge


#Plotting
import os
from pathlib import Path
import seaborn as sns
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
        IV title str: Title for plot
        V filename str | Path: path for saving figure

    Returns:
    ------------
        I   None
    '''
    plt.figure()
    ax = sns.heatmap(MSEs, xticklabels=etas, yticklabels=lams, annot=True, robust = True)
    #sns.title(title)
    ax.set(xlabel="Lambda", ylabel="Etas")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path / filename)

def fit(X, y, grad_cost_func, scheduler, batches = 1, epochs = 100, seed = 13):
    '''
    Description:
    ------------
    Function for fitting regressin parameters.

    Parameters:
    ------------
        I   X (np.ndarray): The design matrix, with n rows of p features each
        II  y (np.ndarray): The target vector, with length n
        III grad_cost_func Callable: gradient of cost function.
        IV batches  int=1: Number of minibatches to split data into. If 1, just normal gradient descent
        V epochs int=100: Number of iterations to do.
        VI seed int=13: Seed to inititalze random numbers.

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
            

# Data
def f(x):
    return -3+5 * x -3*x**2

n_datapoints = 100
x = np.random.rand(n_datapoints, 1)
y = f(x) + 0.3*np.random.randn(n_datapoints, 1)
X = np.c_[np.ones(n_datapoints), x, x**2]

n_etas = 10
n_lams = 9
etas = [10**(-k) for k in range(n_etas)]
lams = [10**(-k) for k in range(n_lams)]
batches = 10
epochs = 20
n_methods = 8

#Keep track of best achieved MSE
MSE_best_by_method = []
MSE_method_names = []

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
print(f"Best by method MSE {np.shape(np.asarray(MSE_best_by_method))}")
MSE_method_names.append("Constant")
print(f"MSE momentum: {MSEs}")
plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")

#Momentum
method_name = "Momemtum"
MSEs = np.zeros((n_etas, n_lams))
for i, eta in enumerate(etas):
    for j, lam in enumerate(lams):
        scheduler = Momentum(eta, 0.1)
        theta = fit(X, y, grad_cost_Ridge_lam(lam), scheduler, epochs=epochs)
        MSEs[i][j] = cost_OLS(y, X, theta)

MSE_best_by_method.append(np.min(MSEs, axis = 0))
MSE_method_names.append("Momemtum")
print(f"MSE momentum: {MSEs}")
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
print(f"MSE momentum: {MSEs}")
plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


#SGD with momemtum
method_name = "Momemtum_SGD"
MSEs = np.zeros((n_etas, n_lams))
for i, eta in enumerate(etas):
    for j, lam in enumerate(lams):
        scheduler = Momentum(eta, 0.1)
        theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
        #print(theta)
        MSEs[i][j] = cost_OLS(y, X, theta)

MSE_best_by_method.append(np.min(MSEs, axis = 0))
MSE_method_names.append("Momemtum SGD")
print(f"MSE momentum: {MSEs}")
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
print(f"MSE momentum: {MSEs}")
plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


#Adagrad with momemtum
method_name = "AdagradMomemtum"
MSEs = np.zeros((n_etas, n_lams))
for i, eta in enumerate(etas):
    for j, lam in enumerate(lams):
        scheduler = AdagradMomentum(eta, 0)
        theta = fit(X, y, grad_cost_Ridge_lam(lam),scheduler, batches=batches, epochs=epochs)
        #print(theta)
        MSEs[i][j] = cost_OLS(y, X, theta)

MSE_best_by_method.append(np.min(MSEs, axis = 0))
MSE_method_names.append("Adagrad mom")
print(f"MSE momentum: {MSEs}")
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
print(f"MSE momentum: {MSEs}")
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
print(f"MSE momentum: {MSEs}")
plot_etas_lams(etas, lams, MSEs, method_name, method_name + ".png")


#Define path for plot
import os
from pathlib import Path
cwd = os.getcwd()
path = Path(cwd) / "FigurePlots"

if not path.exists():
    path.mkdir()

plt.figure()
ax = sns.heatmap(np.asarray(MSE_best_by_method),
                  yticklabels=MSE_method_names, 
                  xticklabels=lams,
                  annot=np.asarray(MSE_best_by_method))
ax.set(xlabel="Gradient descent methods", ylabel="Lambda value")

plt.title("Best MSE error by gradient descent method for Ridge")
#plt.xticks(rotation=-30)
#plt.tight_layout()
plt.savefig(path / "PartARidgeHeatMap.png")
#plt.show()

print(f"Best MSE array: {np.asarray(MSE_best_by_method)}")
'''
#Plot MSE by gradient method
plt.figure()
plt.bar(range(0, 2*len(MSE_method_names), 2), MSE_best_by_method,  tick_label =MSE_method_names)
plt.title("Best MSE error by gradient descent method for Ridge")
plt.xticks(rotation=-30)
plt.tight_layout()
plt.savefig(path / "PartARidge.png")

'''





