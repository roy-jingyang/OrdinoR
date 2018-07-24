#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of the method of mining overlapping
organizational models using Model-based Overlapping Clustering, proposed by
Yang et al. (ref. J.Yang et al., BPM 2018).
'''

# Class definition
class MOC:
    '''
    Class variables:

    n_components: int, defaults to 1.
        The number of components in the model.

    tol: float, defaults to 1e-6.
        The convergence threshold.

    #TODO: for the moment, NO need to perform multiple initialization.
    n_init: int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    max_iter: int, defaults to 10.
        The number of iterative alternating updates to run.

    M_init: array-like, shape (n_samples, n_components), optional
        The user-provided initial membership matrix M, defaults to None.
        If None, random initialization is used.

    '''
    def __init__(self, 
            n_components=1, tol=1e-6, n_init=1, max_iter=10, M_init=None):
        self.n_components = n_components
        self.tol = tol
        self.n_init = n_init
        self.max_iter = max_iter
        self.M_init = M_init
    
    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        C : array-like, shape (n_samples, n_components), component membership
        '''
        from numpy import array, dot, infty
        from numpy.random import randint
        from numpy.linalg import pinv

        k = self.n_components
        if self.M_init is not None:
            M = self.M_init
        else:
            # Initialize a random guess of membership M
            # Random initialization of a binary matrix
            M = randint(2, size=(len(X), k))
            
        iteration = -1
        new_M = None
        current_log_likelihood = None
        delta_log_likelihood = infty
        print('Fitting MOC model:')
        while delta_log_likelihood > self.tol: # while not converged
            if iteration >= self.max_iter:
                print('[Warning] Model did not converged within a max. of ' +
                        '{} iterations (delta= {:.4f}).'.format(max_iter, tol))
                return M

            iteration += 1
            print('Iteration {}:'.format(iteration))
            prev_log_likelihood = current_log_likelihood

            # Alternate between updating M and A

            # update A: assume loss function sqaured Euclidean Distance
            pinv_M = pinv(M) 
            A = dot(pinv_M, X)

            # update M: for each row, apply appropriate search algorithms
            new_M = list()
            for i in range(X.shape[0]):
                m_best = self._enumerate(X[i,:], M[i,:], A, k)
                new_M.append(m_best)
            M = array(new_M)

            # calculate new log-likelihood:
            current_log_likelihood = self.score(X, M, A)
            print('score = {:.3f}'.format(current_log_likelihood))
            if prev_log_likelihood is not None: # if not the initial run
                delta_log_likelihood = \
                        current_log_likelihood - prev_log_likelihood
        print('Model fitted in {} iterations.'.format(iteration))
        return M

    def score(self, X, M, A):
        from numpy import dot, log, power
        #print('Calculating the score: ', end='')
        # calculate alpha_ij
        score_alpha = 0.0
        for j in range(M.shape[1]): # k
            M_j = M[:,j]
            pie_j = (1.0 / X.shape[0]) * len(M_j[M_j == True])
            for i in range(M.shape[0]): # n
                # Bernoulli
                alpha_ij = power(pie_j, M[i,j]) * \
                        power((1 - pie_j), (1 - M[i,j]))
                score_alpha += log(alpha_ij)

        # calculate the Bregman divergence
        score_divergence = 0.0
        from scipy.spatial import distance
        MA = dot(M, A)
        for i in range(X.shape[0]): # n
            for h in range(X.shape[1]): # d
                # Squared Euclidean Distance
                score_divergence += power(
                        distance.euclidean(X[i,h], MA[i,h]), 2)

        score = score_alpha - score_divergence
        #print('{:.4f}'.format(score))
        return score

    def _enumerate(self, x, m0, A, k):
        from numpy import array, dot, power
        from scipy.spatial import distance
        from itertools import combinations

        L0 = distance.euclidean(x, dot(m0, A))
        Lmin = L0
        m_best = m0

        # Constrained (pruned): possible settings sum_{i = 1 to c} C^i_k
        '''
        for i in range(1, 2 + 1):
        '''
        # Exhaustively enumerate all possible settings sum_{i = 1 to k} C^i_k
        for i in range(1, k + 1):
            for index in combinations(range(k), i):
                m = array([0] * k)
                m[[index]] = 1
                L = power(distance.euclidean(x, dot(m, A)), 2)
                if L < Lmin:
                    Lmin = L
                    m_best = m
        '''
        # Disjoint: reduced case
        for i in range(k):
            m = np.array([0] * k)
            m[i] = 1
            L = np.power(distance.euclidean(x, np.dot(m, A)), 2)
            if L < Lmin:
                Lmin = L
                m_best = m
        '''
        return m_best
    
    # TODO
    def _dynamicm(self, x, m0, A, k):
        pass


def mine(profiles,
        n_groups,
        warm_start_input_fn=None):
    '''
    This method implements the algorithm of using Gaussian Mixture Model
    for mining an overlapping organizational model from the given event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        warm_start_input_fn: str, optional
            Filename of the initial guess of clustering. The file should be
            formatted as:
                Group ID, resource; resource; ...
            with each line in the CSV file representing a group.
            The default is None, meaning warm start is NOT used.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''

    print('Applying overlapping organizational model mining using ' + 
            'clustering-based MOC:')
    # step 1. Importing warm-start (initial guess of clustering from file
    moc_warm_start = (warm_start_input_fn is not None)
    if moc_warm_start:
        from numpy import zeros
        from pandas import DataFrame
        m = DataFrame(zeros((len(profiles), n_groups)), index=profiles.index)
        from csv import reader
        with open(warm_start_input_fn, 'r') as f:
            count_groups = 0
            for row in reader(f):
                for r in row[1].split(';'):
                    m.loc[r][count_groups] = 1 # equals to m[i,j]
                count_groups += 1
        if n_groups != count_groups:
            exit('Invalid initial guess detected. Exit with error.')
        else:
            print('Initial guess imported from file "{}".'.format(
                warm_start_input_fn))

    # step 2. Training the model
    if moc_warm_start:
        moc_model = MOC(n_components=n_groups, M_init=m.values)
    else:
        moc_model = MOC(n_components=n_groups)
    mat_membership = moc_model.fit_predict(profiles.values)

    # step 3. Deriving the clusters as the end result
    from numpy import nonzero
    from collections import defaultdict
    og = defaultdict(lambda: set())
    # TODO: more pythonic way required
    for i in range(len(mat_membership)):
        # check if any valid membership exists for the resource based on
        # the results predicted by the obtained MOC model
        if len(nonzero(mat_membership[i,:])[0]) > 0: # valid
            for j in nonzero(mat_membership[i,:])[0]:
                og[j].add(profiles.index[i])
        else: # invalid (unexpected exit)
            exit('[Fatal error] MOC failed to produce a valid result')

    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

