#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# The class definition goes here
class MOC:
    # member variables
    '''
    n_components: int, defaults to 1.
        The number of components in the model.

    tol: float, defaults to 1e-3.
        The convergence threshold.

    TODO: now leave n_init to be 1 (i.e. no multiple initialization to perform)
    n_init: int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    max_iter: int, defaults to 10.
        The number of iterative alternating updates to run.
    '''

    # class methods
    def __init__(self, n_components=1, tol=1e-3, n_init=1, max_iter=10):
        self.n_components = n_components
        self.tol = tol
        self.n_init = n_init
        self.max_iter = max_iter
    
    def fit_predict(self, X):
        '''Fit and then predict labels for data

        Parameters
        __________
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_components], component membership
        '''

        iteration = -1
        # Initialize a random guess of membership M
        k = self.n_components
        rM = np.random.rand(X.shape[0], k)
        M = list()
        for row in rM:
            t = np.random.choice(row)
            M.append(np.array(row >= t))
        M = np.array(M)
        
        new_M = None
        current_log_likelihood = None
        delta_log_likelihood = np.infty
        while delta_log_likelihood > self.tol: # while not converged
            if iteration >= self.max_iter: # max_iter reached
                print('Max. iteration reached. Not converged ' +
                        '(delta ={:.4f})'.format(delta_log_likelihood))
                return M

            iteration += 1
            print('Iteration {}:'.format(iteration))
            prev_log_likelihood = current_log_likelihood

            # Alternate between updating M and A
            # update A: assume loss function Euclidean Distance
            pinv_M = np.linalg.pinv(M) 
            A = np.dot(pinv_M, X)
            # update M: use dynamicM
            new_M = list()
            for i in range(X.shape[0]):
                m_best = self.dynamicM(X[i,:], M[i,:], A, k)
                new_M.append(new_M)
            M = np.array(new_M)
            # calculate new log-likelihood:
            current_log_likelihood = self.score(X, M, A)
            if prev_log_likelihood is not None:
                # delta >= 0
                delta_log_likelihood = \
                        current_log_likelihood - prev_log_likelihood
        return M

    def _dynamicM(x, m0, A, k):
        pass

    def score(X, M, A):
        # calculate alpha_ij
        score_alpha = 0.0
        for j in range(M.shape[1]): # k
            M_j = M[:,j]
            pie_j = (1.0 / X.shape[0]) * len(M_j[M_j == True])
            for i in range(M.shape[0]): # n
                # Bernoulli
                score_alpha += np.log(
                        np.power(pie_j, M[i,j]) + 
                        np.power(1 - pie_j, 1 - M[i,j]))
        # calculate the Bregman divergence
        score_divergence = 0.0
        MA = np.dot(M, A)
        for i in range(X.shape[0]): # n
            for h in range(X.shape[1]): # d
                # Euclidean Distance
                score_divergence += np.log(np.power(X[i,h] - MA[i,h], 2))

        return (score_alpha - score_divergence)

import copy
import matplotlib.pyplot as plt
from collections import defaultdict
#from EvaluationOptions import Unsupervised

#def mine(cases):
if __name__ == '__main__':
    moc = MOC(n_components = 5)
    X = np.random.rand(5, 3)
    moc.fit_predict(X)

