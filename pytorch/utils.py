import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

def plot_classifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '1' and '2'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by D feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.linspace(x1_min, x1_max, 200)
    x2_line =  np.linspace(x2_min, x2_max, 200)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model.predict(mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    plt.figure()
    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, -y_pred.astype(int), # unsigned int causes problems with negative sign... o_O
                cmap=plt.cm.RdBu, alpha=0.6)

    plt.scatter(x1[y==0], x2[y==0], color="b", label="class 0")
    plt.scatter(x1[y==1], x2[y==1], color="r", label="class 1")
    plt.legend()

def euclidean_dist_squared(X_1, X_2):
    """Computes the Euclidean distance between rows of 'X_1' and rows of 'X_2'

    Parameters
    ----------
    X : an N by D tensor # N is number of train example, D is number of feature
    X_2: an T by D tensor # T is number of test example

    Returns: an array of size N by T containing the pairwise squared Euclidean distances."""

    X1_norms_sq = torch.sum(X_1 ** 2, axis=1)
    X2_norms_sq = torch.sum(X_2 ** 2, axis=1)
    dots = X_1 @ X_2.T

    return X1_norms_sq[:, np.newaxis] + X2_norms_sq[np.newaxis, :] - 2 * dots