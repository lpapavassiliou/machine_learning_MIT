"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


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
    # TODO
    
    # import is here because of MIT class evaluation
    from scipy.stats import multivariate_normal
    
    # extract input dimensions
    n,d = X.shape
    K = mixture.p.shape[0]
    
    # build the C list, whose every element C[i] is the array of indices 
    # correspondig to non zero element of X[i]
    C = []
    for i in range(n):
        C.append(np.nonzero(X[i])[0])
    
    # build the prior matrix
    log_prior =  np.tile(np.log(mixture.p), (n, 1))
    
    # build the likelihood matrix
    likelihoods = np.zeros((n,K))
    for i in range(n):
        for j in range(K):
            x_relevant = X[i][C[i]]
            mu_relevant = mixture.mu[j][C[i]]
            cov_matrix = mixture.var[j]*np.eye(x_relevant.shape[0])
            if x_relevant.size > 0:
                likelihoods[i,j] = np.maximum(multivariate_normal(mean=mu_relevant, cov=cov_matrix).pdf(x_relevant), 1e-10)
            else:
                likelihoods[i,j] = 1
    

    # build the f[u,j] matrix
    f = log_prior + np.log(likelihoods)

    # build the posterior matrix
    normalizer = np.sum(np.exp(f), axis=1)
    log_posterior = f - np.tile(np.log(normalizer), (K,1)).T
    post = np.exp(log_posterior)

    # weighted_likelihoods[u][j] represents the probability of picking the model j
    # and then generating the point u
    weighted_likelihoods = np.multiply(likelihoods, mixture.p)
    
    # compute datasetLL
    dataset_LL = np.sum(np.log(np.sum(weighted_likelihoods, axis=1)))
    
    return post, dataset_LL




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
    
    while datasetLL-old_datasetLL >= 1e-6 * np.abs(datasetLL)+0.000001:
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
    raise NotImplementedError
