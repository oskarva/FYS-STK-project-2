"""
This code uses a neural network to predict a good or bad case with the Wisconsin Cancer Data.
"""
import seaborn as sns
from neural_network import FFNN
from cost_functions import *
from schedulers import *
from activation_functions import *
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import preprocessing



np.random.seed(2023)

# Load the Wisconsin cancer data
data = load_breast_cancer()
target = data["target"]
target = np.reshape(target, (target.size, 1))
X = data["data"]

# Scale the data
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

#Crossvalidation
from sklearn.model_selection import KFold
n_folds = 5
kf = KFold(n_splits=n_folds)

# Define neural network
n_featurs = np.shape(X_scaled)[1]
n_etas = 7
etas = np.geomspace(0.0001, 0.1, n_etas)
epochs = 100
batches = 10
lmbd = 0

# Topology of network
n_hidden_neurons = 30
dimensions = (n_featurs, n_hidden_neurons, 1)


# Keep track of performance by eta
MSEs_train = np.zeros((n_folds, n_etas))
accuracies_train = np.zeros((n_folds, n_etas))
MSEs_val = np.zeros((n_folds, n_etas))
accuracies_val = np.zeros((n_folds, n_etas))

# Keep track of predictions and the training and validation sets used
pred_train = []
pred_val = []
data_train = []
data_val = []

#Run through crossvalidation and tuning of eta
for i, (train_index, test_index) in  enumerate(kf.split(X_scaled, target)):
    for j, eta in enumerate(etas):
        NN = FFNN(
            dimensions,
            hidden_func=sigmoid,
            output_func=sigmoid,
            cost_func=CostLogReg,
            seed=13,
        )
        scores = NN.fit(
            X_scaled[train_index],
            target[train_index],
            Adam(eta, 0.9, 0.99),
            batches=batches,
            epochs=epochs,
            lam=lmbd,
            X_val=X_scaled[test_index],
            t_val=target[test_index],
        )

        # Store results
        MSEs_val[i][j] = scores["val_errors"][-1]
        accuracies_val[i][j] = scores["val_accs"][-1]
        MSEs_train[i][j] = scores["train_errors"][-1]
        accuracies_train[i][j] = scores["train_accs"][-1]

        # Predictions
        if i == 0:
            pred_train.append(NN.predict(X_scaled[train_index]))
            pred_val.append(NN.predict(X_scaled[test_index]))
            data_train.append(target[train_index])
            data_val.append(target[test_index])

# The model that predicted best.
print(f"The beset validation acurcary over all folds and etas: {np.max(accuracies_val)}")
print(f"\n The best acuracy for each fold of MSE over eta values: {np.max(accuracies_val, axis = 1)} \n")
print(f"\n The mean acuracy for k folds for different eta values: {np.mean(accuracies_val, axis = 0)} \n")
print(f"The best val accuracy over folds: {np.max(np.mean(accuracies_val, axis = 0))}")

best_index = np.argmax(np.mean(accuracies_val, axis = 0))


#Same test but now with sk-learn
from sklearn.neural_network import MLPClassifier

DNN_scikit = np.zeros((n_folds, n_etas), dtype=object)
DNN_score = np.zeros((n_folds, n_etas))

for i, (train_index, test_index) in  enumerate(kf.split(X_scaled, target)):
    for j, eta in enumerate(etas):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=10000)
        dnn.fit(X_scaled[train_index], target[train_index].ravel())
        
        DNN_scikit[i][j] = dnn
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_scaled[test_index], target[test_index]))
        DNN_score[i][j] = dnn.score(X_scaled[test_index], target[test_index])
        print()

#print(np.mean(DNN_score, axis=0))
# Define path for saving figure
from pathlib import Path
import os

cwd = os.getcwd()
path = Path(cwd) / "FigurePlots" / "Cancer_data"
if not path.exists():
    path.mkdir()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Plot confusion matrix for validation set
confusion_mat = confusion_matrix(
    data_val[best_index].flatten(), pred_val[best_index].flatten(), normalize="true"
)

plt.figure()
sns.heatmap(confusion_mat, annot=True)
plt.title("Confusion matrix cancer data. \n Validation set for neural network")
plt.savefig(path / "neural_net_confusion.png")


# Plot confusion matrix for training set
confusion_mat = confusion_matrix(
    data_train[best_index].flatten(), pred_train[best_index].flatten(), normalize="true"
)

plt.figure()
sns.heatmap(confusion_mat, annot=True)
plt.title("Confusion matrix cancer data. \n Training set for neural network")
plt.savefig(path / "neural_net_confusion_train.png")
plt.show()


DNN_score_mean = np.mean(DNN_score, axis=0)
accuracies_val_mean = np.mean(accuracies_val, axis = 0)


print(f"\n Acuraccy by eta values, average over folds. Sklearn: {DNN_score_mean}")
print(f"\n Acuraccy by eta values, average over folds. Own neural network: {accuracies_val_mean}")
print(f"\n The best achieved accuracy by sklearn was {np.max(DNN_score_mean)}")
print(f"\n The best achieved accuracy by neural network was {np.max(accuracies_val_mean)}")



