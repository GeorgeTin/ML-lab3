import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random


# NOTE: you do not need to handle the W argument for this part!
# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    # TODO: Ass 2
    # TODO: compute the values of prior for each class!
    # ==========================

    for jdx, cls in enumerate(classes):

        idx = np.where(labels == cls)[0]
        nk = len(idx)

        prior[jdx] = nk/Npts

    # ==========================

    return prior


# NOTE: you do not need to handle the W argument for this part!
# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)

def mlParams(X, labels, W = None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: Ass 1
    # TODO: fill in the code to compute mu and sigma!
    # ==========================

    for jdx, cls in enumerate(classes):

        idx = np.where(labels == cls)[0]
        xlc = X[idx, :]

        # Compute mu
        mu[jdx] = np.sum(xlc, axis=0) / len(xlc)

        # Compute sigma
        matrix_substraction = (xlc - mu[jdx].transpose()).transpose()

        for j in range(len(matrix_substraction)):
            matrix_substraction[j] = [x * x for x in matrix_substraction[j]]

        sigma_sum = np.sum(matrix_substraction, axis=1)
        sigma[jdx] = np.diag(sigma_sum / len(xlc))

    # ==========================

    return mu, sigma

