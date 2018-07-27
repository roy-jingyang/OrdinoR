# -*- coding: utf-8 -*-

# TODO: in the current implementation, we assume a Gaussian distribution for
# the latent categories of the data samples, thus the Bregman divergence is the
# sqaure Euclidean distance.
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

        if self.M_init is not None:
            M = self.M_init
        else:
            # Initialize a random guess of membership M
            # Random initialization of a binary matrix
            M = randint(2, size=(len(X), self.n_components))
            
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
                #m_best = self._enumerate(X[i,:], M[i,:], A)
                m_best = self._dynamicm(X[i,:], M[i,:], A) #TODO
                new_M.append(m_best)
            M = array(new_M)

            # calculate new log-likelihood:
            current_log_likelihood = self.score(X, M, A)
            print('score = {:.3f}'.format(current_log_likelihood))
            if prev_log_likelihood is not None: # if not the initial run
                delta_log_likelihood = (current_log_likelihood
                        - prev_log_likelihood)
        print('Model fitted in {} iterations.'.format(iteration))
        return M

    # Definition of the likelihood function: log p(X, M, A) to be maximized
    def score(self, X, M, A):
        from numpy import dot, log, power
        #print('Calculating the score: ', end='')
        # calculate alpha_ih
        score_alpha = 0.0
        for h in range(M.shape[1]): # n_components
            M_h = M[:,h]
            pie_h = (1.0 / X.shape[0]) * len(M_h[M_h == True])
            for i in range(M.shape[0]): # n_samples
                # Bernoulli
                alpha_ih = (power(pie_h, M[i,h])
                        * power((1 - pie_h), (1 - M[i,h])))
                score_alpha += log(alpha_ih)

        # calculate the Bregman divergence
        # currently Squared Euclidean Distance
        score_divergence = 0.0
        from scipy.spatial.distance import sqeuclidean
        MA = dot(M, A)
        for i in range(X.shape[0]): # n_components
            for j in range(X.shape[1]): # n_features
                score_divergence += sqeuclidean(X[i,j], MA[i,j])

        score = score_alpha - score_divergence
        #print('{:.4f}'.format(score))
        return score

    # search over all possible settings in the naive way
    def _enumerate(self, x, m0, A):
        from numpy import array, dot
        from scipy.spatial.distance import sqeuclidean
        from itertools import combinations

        L0 = sqeuclidean(x, dot(m0, A))
        Lmin = L0
        k = len(m0)
        m_best = m0

        # Constrained (pruned): possible settings sum_{i = 1 to c} C^i_k
        '''
        for i in range(1, 2 + 1):
        '''
        # Exhaustively enumerate all possible settings sum_{i = 1 to k} C^i_k
        # start from 1 ensures that at least one scalar with value True
        for i in range(1, k + 1):
            for index in combinations(range(k), i):
                m = array([0] * k)
                m[[index]] = 1
                L = sqeuclidean(x, dot(m, A))
                if L < Lmin:
                    Lmin = L
                    m_best = m
        '''
        # Disjoint: reduced case
        for i in range(k):
            m = np.array([0] * k)
            m[i] = 1
            L = sqeuclidean(x, np.dot(m, A))
            if L < Lmin:
                Lmin = L
                m_best = m
        '''
        return m_best
    
    # search on each separate threads with different initial setting
    # greedily proceed on the fly (idea similar to DP)
    def _dynamicm(self, x, m0, A):
        from threading import Thread
        from numpy import array, dot, infty, where
        from scipy.spatial.distance import sqeuclidean
        from copy import copy
        class SearchThread(Thread):
            def __init__(self, x, m, A):
                Thread.__init__(self)
                self.x = x
                self.A = A
                self.m = m # the guess (per thread)
                self.L = sqeuclidean(x, dot(m, A))
            def run(self):
                # TODO
                # no need for any messy procedure!
                # just:
                # (1) start with the local copy (with 1 cluster turned on)
                # (2) generate the combinations as candidate positions
                # (3) evaluate by level, with 2 turned on, then 3 ...
                # (4) at each level, see if better results come out
                #       continue to next level if there are;
                #       return if NOT!
                # Let's just do this after the test.
                while True:
                    # if the thread is still active:
                    # find the next best cluster to be "turned on", i.e.
                    # the one resulting in smaller value of L
                    next_on = None
                    next_min_L = infty
                    for i in where(m == 0):
                        cand_m = copy(self.m)
                        cand_m[i] = 1
                        cand_L = sqeuclidean(self.x, dot(cand_m, self.A))
                        if cand_L < next_min_L:
                            next_min_L = cand_L
                            next_on = i
                    if next_min_L < self.L:
                        # update the guess if turnining on a next cluster
                        # makes the loss smaller
                        self.L = next_min_L
                        self.m[next_on] = 1
                    else:
                        return

        k = len(m0)
        separate_search_threads = list()
        for h in range(k):
            m = array([0] * k)
            m[h] = 1
            s = SearchThread(x, m, A)
            s.start()
            separate_search_threads.append(s)
        L0 = sqeuclidean(x, dot(m0, A))
        m_best = m0
        for h in range(k):
            separate_search_threads[h].join()
            if separate_search_threads[h].L < L0:
                L0 = separate_search_threads[h].L
                m_best = separate_search_threads[h].m

        return m0

