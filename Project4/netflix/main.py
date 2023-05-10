import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")           # nxd matrix
X = np.loadtxt("netflix_incomplete.txt") # nxd matrix

# TODO: Your code here

# Pick a set of values for K (number of clusters) and seed (random seed initialization)
# Try all and pick the best seed
candidates_K = np.array([1,12])
candidates_seed = np.array([0,1,2,3,4])

# Initialize the k-means algorithm providing data, number of clusters and random seed for initial clusters
# Note that init(X,K) returns a K-component mixture model with means, variances and mixing proportions. 
# The K-means algorithm will only care about the means

n = X.shape[0]
datasetLL = 0
models = []   # keeps track of the models and the costs associated with every choice of K

for idx, K in enumerate(candidates_K):
    print('fitting model for K = ' +str(K) + ':')
    best_gauss_mixt = None
    best_posterior = None
    best_seed = None
    max_datasetLL = np.inf
    
    for seed in candidates_seed:
        print('   new seed')
        # initialize the mixture and the posterior model probabilities matrix
        init_gauss_mixt, posterior = common.init(X, K, seed)
        
        # run the algorithm to fit the current model
        # choose the type of model among {kmeans, nnaive_em}
        post_gauss_mixt, posterior, datasetLL = em.run(X, init_gauss_mixt, posterior)
        
        # update best clustering
        if datasetLL < max_datasetLL:
            max_datasetLL = datasetLL
            best_gauss_mixt = post_gauss_mixt
            best_posterior = posterior
            best_seed = seed
            
    # memorize cost and model associated with the current choice of K (and best seed)
    models.append((best_gauss_mixt, best_posterior, max_datasetLL, best_seed))
    

# plot datasetLL and clustering figure of each model
print('\nDatasetLL for each choice of K:')
for idx, (gauss_mixt, posterior, datasetLL, seed) in enumerate(models):
    print('    K=' + str(candidates_K[idx]) + '  datasetLL=' + str(datasetLL))
    #common.plot(X, gauss_mixt, posterior, 'K = '+str(idx+1))

# find best K with BIC method
max_BIC = -np.inf
best = None
print('\nBIC for each choice of K:')
for idx, (gauss_mixt, posterior, datasetLL, seed) in enumerate(models):
    curr_BIC = common.bic(X, gauss_mixt, datasetLL)
    print('    K=' + str(candidates_K[idx]) + '  bic=' + str(curr_BIC))
    if curr_BIC > max_BIC:
        max_BIC = curr_BIC
        best_K = idx+1




