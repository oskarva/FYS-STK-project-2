'''
This code uses neural networks to model a dataset generated by a second degree polynomial.
It tests 3 different activation functions and compares the results.
'''

from neural_network import FFNN
from cost_functions import *
from schedulers import *
from activation_functions import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#Define path for saving figures.
import os
from pathlib import Path
cwd = os.getcwd()
path = Path(cwd) / "FigurePlots" / "B_NeuralNetwork"

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
        III title str: Title for plot
        IV filename str | Path: path for saving figure

    Returns:
    ------------
        I   None
    '''
    plt.figure()
    plt.plot(etas, MSEs)
    plt.xscale("log")
    plt.title("Validation MSE for method" + title)
    plt.tight_layout()
    plt.savefig(path / filename)



#Define data
def f(x):
    return -3+5 * x -3*x**2

n_datapoints = 100
x = np.random.rand(n_datapoints, 1)
y = f(x) + 0.3*np.random.randn(n_datapoints, 1)

#Split in train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Define hyperparameters
n_etas = 10
etas = np.geomspace(10**(-7), 1, n_etas)
n_lams = 10
lams = np.geomspace(10**(-7), 1, n_lams)
batches = 10
epochs = 20
seed = 13

# Define topology of neural network. 1 Hidden layer with 10 neurons.
dimensions = (1, 10, 1)

#Keep track of best MSE by acctivation method
MSE_method = []
method_names =  [] 


#Neural network with activation function sigmoid.
method_name= "Sigmoid"
MSEs = []
for eta in etas:
    NN = FFNN(dimensions, hidden_func=sigmoid, output_func=identity, cost_func=CostOLS, seed=seed)
    scores = NN.fit(x_train, y_train, RMS_prop(eta, 0.9), batches=batches, epochs=epochs, X_val = x_test, t_val = y_test)
    MSEs.append(np.min(scores["val_errors"]))
plot_etas(etas, MSEs, method_name, method_name + ".png")

MSE_method.append(min(MSEs))
method_names.append(method_name)
print(f"\n Best MSE train sigmoid is {np.min(MSEs)}")

#Neural network with activation function RELU
method_name= "RELU"
MSEs = []
for eta in etas:
    NN = FFNN(dimensions, hidden_func=RELU, output_func=identity, cost_func=CostOLS, seed=seed)
    scores = NN.fit(x, y, RMS_prop(eta, 0.9), batches=batches, epochs=epochs, X_val = x_test, t_val = y_test)
    MSEs.append(np.min(scores["val_errors"]))
plot_etas(etas, MSEs, method_name, method_name + ".png")
MSE_method.append(min(MSEs))
method_names.append(method_name)
print(f"\n Best MSE  with RELU is {np.min(MSEs)}")

#Neural network with activation function Leaky RELU
method_name= "Leaky_RELU"
MSEs = []
for eta in etas:
    NN = FFNN(dimensions, hidden_func=LRELU, output_func=identity, cost_func=CostOLS, seed=seed)
    scores = NN.fit(x, y, RMS_prop(eta, 0.9), batches=batches, epochs=epochs, X_val = x_test, t_val = y_test)
    MSEs.append(np.min(scores["val_errors"]))

plot_etas(etas, MSEs, method_name, method_name + ".png")
MSE_method.append(min(MSEs))
method_names.append(method_name)
print(f"\n Best MSE train with Leaky RELU is {np.min(MSEs)}")



#Do a regular OLS for benchmarking
method_name = "OLS_regression"
from OLS_regression import ols_regression

X_train = np.c_[np.ones(x_train.size), x_train, x_train**2]
X_test = np.c_[np.ones(x_test.size), x_test, x_test**2]
pred_train, pred_val, thetas = ols_regression(X_train, X_test, y_train)

MSE_method.append(CostOLS(y_test)(pred_val))
method_names.append(method_name)



#Define new folder for saving figure
path = Path(cwd) / "FigurePlots" / "Neural_Network_By_method"
if not path.exists():
    path.mkdir()


#Plot bar chart of best MSE by activation function
plt.figure()
plt.bar(range(0, 2*len(method_names), 2), MSE_method,  tick_label =method_names)
plt.title("Neural network best MSE val. error by hidden activation")
plt.xticks(rotation=-30)
plt.tight_layout()
plt.savefig(path / "activation_func.png")


