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
    def __init__(self, 
            init_M=None, n_components=1, tol=1e-8, n_init=1, max_iter=100):
        self.n_components = n_components
        self.tol = tol
        self.n_init = n_init
        self.max_iter = max_iter
        self.init_M = init_M
    
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
        k = self.n_components
        if self.init_M is not None:
            M = self.init_M
        else:
            # Initialize a random guess of membership M
            # Random initialization
            rM = np.random.rand(X.shape[0], k)
            M = list()
            for row in rM:
                t = np.random.choice(row)
                #t = sorted(row, reverse=True)[2] # Constrained
                #t = np.amax(row) # Disjoint: reduced case
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
            #print('Iteration {}:'.format(iteration))
            prev_log_likelihood = current_log_likelihood

            # Alternate between updating M and A
            # update A: assume loss function sqaured Euclidean Distance
            pinv_M = np.linalg.pinv(M) 
            A = np.dot(pinv_M, X)
            # update M: use dynamicM
            new_M = list()
            for i in range(X.shape[0]):
                m_best = self._dynamicM(X[i,:], M[i,:], A, k, True)
                new_M.append(m_best)
            M = np.array(new_M)
            # calculate new log-likelihood:
            current_log_likelihood = self.score(X, M, A)
            if prev_log_likelihood is not None:
                # delta >= 0
                delta_log_likelihood = \
                        current_log_likelihood - prev_log_likelihood
                #print('delta = {:.4f}'.format(delta_log_likelihood))
        return M

    def _dynamicM(self, x, m0, A, k, is_exhaustive=False):
        from scipy.spatial import distance
        L0 = distance.euclidean(x, np.dot(m0, A))
        Lmin = L0
        m_best = m0

        if is_exhaustive:
            from itertools import combinations
            # Constrained (pruned): possible settings sum_{i = 1 to c} C^i_k
            # WABO: round = ceil = 4
            # BPIC Open: round = 1, ceil = 2
            # BPIC Closed: round = 1, ceil = 2
            '''
            for i in range(1, 2 + 1):
            '''
            # Run over all possible settings sum_{i = 1 to k} C^i_k
            for i in range(1, k + 1):
                for index in combinations(range(k), i):
                    m = np.array([0] * k)
                    m[[index]] = 1
                    L = np.power(distance.euclidean(x, np.dot(m, A)), 2)
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
        else:
            pass

    def score(self, X, M, A):
        #print('Calculating the score: ', end='')
        # calculate alpha_ij
        score_alpha = 0.0
        for j in range(M.shape[1]): # k
            M_j = M[:,j]
            pie_j = (1.0 / X.shape[0]) * len(M_j[M_j == True])
            for i in range(M.shape[0]): # n
                # Bernoulli
                alpha_ij = np.power(pie_j, M[i,j]) * \
                        np.power((1 - pie_j), (1 - M[i,j]))
                score_alpha += np.log(alpha_ij)
        # calculate the Bregman divergence
        score_divergence = 0.0
        from scipy.spatial import distance
        MA = np.dot(M, A)
        for i in range(X.shape[0]): # n
            for h in range(X.shape[1]): # d
                # Squared Euclidean Distance
                score_divergence += np.power(
                        distance.euclidean(X[i,h], MA[i,h]), 2)

        score = score_alpha - score_divergence
        #print('{:.4f}'.format(score))
        return score

import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from EvaluationOptions import Unsupervised

#def mine(cases):
def mine(cases, num_c):
    print('Applying Model-based Overlapping Clustering:')

    # constructing the performer-activity matrix from event logs
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = []

    for caseid, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            if activity in activity_index:
                pass
            else:
                activity_index.append(activity)
            counting_mat[resource][activity] += 1

    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count

    # logrithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    #k_cluster = 5
    k_cluster = int(num_c)
    # Read in initialized membership if warm start
    #print('Filename for warm-start input (enter if none):', end=' ')
    #f_membership = input()
    f_membership = ''
    if f_membership is not '':
        mat_membership = np.zeros((len(resource_index), k_cluster))
        with open(f_membership, 'r') as f:
            for line in f:
                line = line.strip()
                i = resource_index.index(line.split(',')[0])
                j = int(line.split(',')[1])
                mat_membership[i,j] = True
        moc = MOC(n_components=k_cluster, init_M = mat_membership)
    else:
        moc = MOC(n_components=k_cluster)

    determined = moc.fit_predict(profile_mat)
    # shape: (n_samples,)
    labels = list()
    for row in determined:
        labels.append(tuple(l for l in np.nonzero(row)[0]))
    labels = np.array(labels).ravel()

    # evaluation (unsupervised) goes here
    # only account for valid solution
    is_overlapped = type(labels[0]) != np.int64
    if is_overlapped or len(np.unique(labels)) > 1:
        print('Warning: ONLY VALID solutions are accounted.')
        total_within_cluster_var = Unsupervised.within_cluster_variance(
                profile_mat, labels, is_overlapped)
        #silhouette_score = Unsupervised.silhouette_score(
        #        profile_mat, labels, is_overlapped)
        silhouette_score = -1
        solution = (labels, total_within_cluster_var, silhouette_score)

        #print('score: var(k) = {:.3f}'.format(solution[1]))
        #print('score: silhouette = {:.3f}'.format(solution[2]))

        entities = defaultdict(lambda: set())
        for i in range(len(solution[0])):
            if is_overlapped: # overlapped
                for l in solution[0][i]:
                    entity_id = l
                    entities[l].add(resource_index[i])
            else:
                # only 1 cluster assigned for each
                entity_id = int(solution[0][i])
                entities[entity_id].add(resource_index[i])

        print('Clustering results are ' + \
                ('overlapped.' if is_overlapped else 'disjoint.'))
        print('{} organizational entities extracted.'.format(len(entities)))
        return copy.deepcopy(entities)
    else:
        print('Unexpected solution: is_overlapped = {}'.format(is_overlapped))
        exit()


