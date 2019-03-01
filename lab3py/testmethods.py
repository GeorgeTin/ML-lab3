import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

# ass 4
def mlParams1(X, labels, W = None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    # TODO: Ass 4
    # TODO: fill in the code to compute mu and sigma!
    # ==========================

    for jdx, cls in enumerate(classes):

        idx = np.where(labels == cls)[0]
        xlc = X[idx, :]
        wlc = W[idx, :]
        my_x = np.zeros((Nclasses, Ndims))

        for i in range(len(xlc)):
            for j in range(len(xlc[0])):
                my_x[jdx][j] += (xlc[i][j] * wlc[i])

        wlc_sum = np.sum(wlc)
        print(my_x)

        # Compute mu
        mu[jdx] = np.sum(my_x, axis = 0)/wlc_sum

        print(mu)

        # Compute sigma
        sigma_sum = np.zeros(Ndims)

        for m in range(Ndims):
            for i in range(len(xlc)):
                # for j in range(len(xlc[0])):
                sigma_sum[m] += wlc[i] * pow(xlc[i][m] - mu[jdx][m], 2)
        print(sigma_sum)
        print(wlc_sum)
        print(sigma_sum/wlc_sum)

        sigma[jdx] = np.diag(sigma_sum/wlc_sum)

    # ==========================

    return mu, sigma

# ex 1
def mlParams(X, labels, test, W = None):
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
        print(mu)

        # Compute sigma
        sigma_sum = np.zeros(Ndims)

        for m in range(Ndims):
            for i in range(len(xlc)):
                sigma_sum[m] += pow(xlc[i][m] - mu[jdx][m], 2)

        sigma[jdx] = np.diag(sigma_sum/len(xlc))

    # ==========================

    return mu, sigma

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
        wkl = W[idx, ]
        prior[jdx] = sum(wkl)/sum(W)

    # ==========================
    print(prior)

    return prior

def ml():

    xlc = np.array([[1, 3, 5], [2, 1, 3]])
    muk = np.array([(5, 5, 5)])

    sigma_sum = np.array([0,0,0])

    for m in range(3):
        for i in range(2):
            sigma_sum[m] += pow(xlc[i][m] - muk[0][m], 2)

    print(sigma_sum)

    # print(np.diag(sigma_sum/len(xlc)))

    new_matrix = (xlc - muk[0].transpose()).transpose()

    # print(new_matrix)

    for j in range(len(new_matrix)):
        new_matrix[j] = [x*x for x in new_matrix[j]]

    # print(new_matrix)

    sigma_line = np.sum(new_matrix, axis=1)
    print(sigma_line)
ml()

# labels = np.array([1, 1, 2])
#
# X = np.array(([12, 2],
#               [5, 2],
#               [12, 23]))
#
# mlParams1(X, labels)

# computePrior(labels)