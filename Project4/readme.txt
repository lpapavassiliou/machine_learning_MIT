The task is to build a mixture model for collaborative filtering. You are given a data matrix containing movie 
ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has 
rated only a small fraction of the movies so the data matrix is only partially filled. 
The goal is to predict all the remaining entries of the matrix.

I will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a 
sample from a mixture model. In other words, we have  possible types of users and, in the context of each user, 
we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. 
We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed 
rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently 
re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict
values for all the missing entries in the data matrix.

The folder contains the following files:

    Python files

    - kmeans where we have implemented a baseline using the K-means algorithm
    - naive_em.py where you will implement a first version of the EM algorithm (tabs 3-4)
    - em.py where you will build a mixture model for collaborative filtering (tabs 7-8)
    - common.py where you will implement the common functions for all models (tab 5)
    - main.py where you will write code to answer the questions for this project
    - test.py where you will write code to test your implementation of EM for a given test case

    Data files

    - toy_data.txt a 2D dataset that you will work with in tabs 2-5
    - netflix_incomplete.txt the netflix dataset with missing entries to be completed
    - netflix_complete.txt the netflix dataset with missing entries completed
    - test_incomplete.txt a test dataset to test for you to test your code against our implementation
    - test_complete.txt a test dataset to test for you to test your code against our implementation
    - test_solutions.txt a test dataset to test for you to test your code against our implementation
