"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture

from tqdm import tqdm


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        mask = (X[i, :] != 0)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        total = logsumexp(post[i, :])
        post[i, :] = post[i, :] - total
        ll += total

    return np.exp(post), ll

def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return log_prob


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # TODO
    
    # extract input dimensions
    n, d = X.shape
    K = post.shape[1]

    # build the C list, whose every element C[i] is the array of indices 
    # correspondig to non zero element of X[i]
    C = []
    for i in range(n):
        C.append(np.nonzero(X[i])[0])
    
    # compute delta matrix. delta[l,i]=1 if lth component of X[i] is not 0
    delta = np.zeros((d,n))
    for i in range(n):
        delta[C[i],i] = np.ones(C[i].shape[0])

    # compute the means
    mu = np.zeros((K,d))
    for j in range(K):
        for l in range(d):
            if post[:,j]@delta[l,:] >= 1:
                weights = np.multiply(post[:,j], delta[l,:])
                mu[j,l] = (weights @ X[:,l])/np.sum(weights)
            else:
                mu[j,l] = mixture.mu[j,l]
    
            
    # compute the norms squared
    norms2 = np.zeros((n,K))
    for i in range(n):
        for j in range(K):
            x_relevant = X[i][C[i]]
            mu_relevant = mu[j][C[i]]
            norms2[i,j] = np.linalg.norm(x_relevant-mu_relevant)**2

    
    # build vector of sizes of C[i]
    sizes_C = np.zeros(n)
    for i in range(n):
        sizes_C[i] = C[i].shape[0]
        
    # compute the normalizer of the variances
    normalizer = 1/(post.T @ sizes_C)
    
    # compute the unscaled variances
    unscaled_variances = np.sum(np.multiply(post, norms2), axis=0)
    
    # compute the variances
    var_no_thrsh = np.multiply(normalizer, unscaled_variances)
    var = np.maximum(var_no_thrsh, 0.25)
    
    # compute p
    p = (1/n)*np.sum(post, axis=0)
    
    # create and return gaussian mixture
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    # TODO
    
    # perform first iteration to initialize datasetLL
    post, datasetLL = estep(X, mixture)
    mixture = mstep(X, post, mixture)
    
    # initilialize old_dataset to -inf to make sure we enter the while
    old_datasetLL = -np.inf
    print('     ' + str(datasetLL-old_datasetLL))
    
    # keep updating until the dataset likelihood converges
    
    while datasetLL-old_datasetLL >= 1e-6 * np.abs(datasetLL):
        old_datasetLL = datasetLL
        post, datasetLL = estep(X, mixture)
        mixture = mstep(X, post, mixture)
        print('     ' + str(datasetLL-old_datasetLL))
        
    return mixture, post, datasetLL


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    X_pred = X.copy()
    K, _ = mixture.mu.shape

    for i in range(n):
        mask = X[i, :] != 0
        mask0 = X[i, :] == 0
        post = np.zeros(K)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[j] = np.log(mixture.p[j]) + log_likelihood
        post = np.exp(post - logsumexp(post))
        X_pred[i, mask0] = np.dot(post, mixture.mu[:, mask0])
    return X_pred

