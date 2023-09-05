import numpy as np
from scipy.stats import multivariate_normal

A = np.array([[0.5, 1],
              [0,  -2],
              [1, -1]])
    
C = []
for u in range(3):
    C.append(np.nonzero(A[u])[0])

print(A[1][C[1]])
    
w = np.array([1,2,0])
k = np.array([1,0,-1,0])
# print(k[k!=0])
#print(np.multiply(A.T, w).T)
print(np.maximum(k, 0.5))


cov_matrix = w[:, None, None] * np.eye(1)
print(cov_matrix)