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

def compute_prior_test_method():

    X = np.array((1,5,2))

    sigma = np.array([5, 0, 0], [0, 3, 0], [0, 0, 9])

    log_sigma = np.log(np.linalg.norm(sigma[i]))
    diag_sigma = np.diag(sigma)
    sigma_k = np.diag(1 / diag_sigma)

    for j in range(Npts):
        log_prior = np.log(prior[i])

        # TODO: continua aici: merge dot product intre diag_sigma si ceilalti vectori??
        # for k in range(len(X[j])):
        #     product_val[k] *= 1/sigma[i][k][k]

        product_val = product_val.dot((X[j] - mu[i]).T)

        logProb[i][j] = -(1 / 2) * log_sigma - (1 / 2) * product_val + log_prior


compute_prior_test_method()
# labels = np.array([1, 1, 2])
#
# X = np.array(([12, 2],
#               [5, 2],
#               [12, 23]))
#
# mlParams1(X, labels)

# computePrior(labels)