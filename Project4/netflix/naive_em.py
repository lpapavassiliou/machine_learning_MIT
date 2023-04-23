"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import *



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        post: np.ndarray (n, K) holding the soft counts
            for all components for all examples
        dataset_log_likelihood: float holding log-likelihood of the assignment
    """
    # my code
    
    # import is here because of MIT class evaluation
    from scipy.stats import multivariate_normal
    
    # recover input dimensions
    n,d = X.shape
    K = mixture.mu.shape[0]
    
    # likelihoods[i][j] represents the pdf of gaussian j evaluated in point i
    likelihoods = np.zeros((n,K))
    for j in range(K):
        likelihoods[:,j] = multivariate_normal(mean=mixture.mu[j], cov=mixture.var[j]*np.eye(d)).pdf(X)
    
    # weighted_likelihoods[i][j] represents the probability of picking the model j
    # and then generating the point i
    weighted_likelihoods = np.multiply(likelihoods, mixture.p)
    
    # normalize the weighted_linkelihoods so that each association i-j is weighted 
    # according to the probability that point i was generated by the other models
    normalizers = np.sum(weighted_likelihoods, axis=1)
    post = np.multiply(weighted_likelihoods.T, 1/normalizers).T
    
    # dataset_log_likelihood is the sum over all the points of their log-probability 
    # of being generated by the mixture gaussian model
    dataset_log_likelihood = np.sum(np.log(np.sum(weighted_likelihoods, axis=1)))
    return post, dataset_log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # TODO
    
    # extract the dimensions
    n, K = post.shape
    d = X.shape[1]
    
    # compute the priors
    n_hat = np.sum(post, axis=0)
    p = n_hat/n
    
    # compute the means. For each model j, the mean mu_j is obtained by multiplying 
    # each raw vector of X by the corrisponding element of post[:,j], and then 
    # summing over the columns. Then normalize everything by the sum of the posteriors relative to j
    mu = np.zeros((K,d))
    for j in range(K):
        mu[j,:] = np.sum( (np.multiply(X.T, post[:,j]).T) , axis=0)/n_hat[j]
    
    # compute norms  ||x_i - mu_j||^2
    norms = np.zeros((n, K))
    for i in range(n):
        for j in range(K):
            norms[i, j] = np.linalg.norm(X[i] - mu[j])**2
            
    # compute variances
    unscaled_var = np.multiply(norms, post).sum(axis=0)
    var = np.multiply( (1/(n_hat*d)), unscaled_var)

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
    mixture = mstep(X, post)
    
    # initilialize old_dataset to -inf to make sure we enter the while
    old_datasetLL = -np.inf
    
    # keep updating until the dataset likelihood converges
    while datasetLL-old_datasetLL >= 1e-6 * np.abs(datasetLL):
        old_datasetLL = datasetLL
        post, datasetLL = estep(X, mixture)
        mixture = mstep(X, post)
        
    return mixture, post, datasetLL