import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if len(y)==0:
        return -1
    else:
        return stats.mode(y.flatten())[0][0]

def euclidean_dist_squared(X_1, X_2):
    """Computes the Euclidean distance between rows of 'X_1' and rows of 'X_2'

    Parameters
    ----------
    X : an N by D numpy array # N is number of train example, D is number of feature
    X_2: an T by D numpy array # T is number of test example

    Returns: an array of size N by T containing the pairwise squared Euclidean distances.

    Python/Numpy (and other numerical languages like Matlab and R)
    can be slow at executing operations in `for' loops, but allows fast
    hardware-dependent vector and matrix operations. By taking advantage of SIMD
    registers and multiple cores (and faster matrix-multiplication algorithms),
    vector and matrix operations in Numpy will often be several times faster
    than if you implemented them yourself in a fast language like C. The
    following code will form a matrix containing the squared Euclidean
    distances between all training and test points. If the output is stored in
    D, then element D[i,j] gives the squared Euclidean distance between training
    point i and testing point j. It exploits the identity (a-b)^2 = a^2 + b^2 - 2ab.
    The right-hand-side of the above is more amenable to vector/matrix operations.
    """
    # for reference, sklearn.metrics.pairwise.euclidean_distances
    # does this but a little bit nicer; this code is just here so you can
    # easily see that it's not doing anything actually very complicated

    X1_norms_sq = np.sum(X_1 ** 2, axis=1)
    X2_norms_sq = np.sum(X_2 ** 2, axis=1)
    dots = X_1 @ X_2.T

    return X1_norms_sq[:, np.newaxis] + X2_norms_sq[np.newaxis, :] - 2 * dots