# -*- coding: utf-8 -*- # TODO: in the current implementation, we assume a Gaussian distribution for
# the latent categories of the data samples, thus the Bregman divergence is the
# square Euclidean distance.
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

    is_disjoint: Boolean, defaults to False
        The Boolean flag indicating whether a disjoint result (the reduced
        case) is required.

    '''
    def __init__(self, 
            n_components=1, tol=1e-6, n_init=1, max_iter=100, M_init=None,
            #is_disjoint=True):
            is_disjoint=False):
        self.n_components = n_components
        self.tol = tol
        self.n_init = n_init
        self.max_iter = max_iter
        self.M_init = M_init
        self.is_disjoint = is_disjoint
    
    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        C : array-like, shape (n_samples, n_components), component membership
        '''
        from numpy import array, dot, infty, zeros
        from numpy.random import randint
        from numpy.linalg import pinv

        if self.M_init is not None:
            M = self.M_init
        else:
            # Initialize a random guess of membership M
            # Random initialization of a binary matrix
            if self.is_disjoint:
                M = zeros((len(X), self.n_components))
                for row in M:
                    row[randint(self.n_components)] = 1 # only 1
            else:
                M = randint(2, size=(len(X), self.n_components)) # arbitrary
            
        iteration = -1
        new_M = None
        current_log_likelihood = None
        delta_log_likelihood = infty
        print('Fitting MOC model:')
        while delta_log_likelihood > self.tol: # while not converged
            if iteration >= self.max_iter:
                print('[Warning] Model did not converged within a max. of ' +
                        '{} iterations (delta= {:.4f}).'.format(
                            self.max_iter, self.tol))
                return M

            iteration += 1
            print('Iteration {}:'.format(iteration))
            prev_log_likelihood = current_log_likelihood

            # Alternate between updating M and A

            # update A: assume loss function sqaured Euclidean Distance
            pinv_M = pinv(M) 
            A = dot(pinv_M, X)
            #print(A)

            # update M: for each row, apply appropriate search algorithms
            new_M = list()
            for i in range(X.shape[0]):
                #m_best = self._enumerate(X[i,:], M[i,:], A)
                m_best = self._dynamicm(X[i,:], M[i,:], A)
                new_M.append(m_best)
            M = array(new_M)
            #print(M)

            # calculate new log-likelihood:
            current_log_likelihood = self.score(X, M, A)
            print('score = {}'.format(current_log_likelihood))
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
        if self.is_disjoint:
            for i in range(k):
                m = np.array([0] * k)
                m[i] = 1
                L = sqeuclidean(x, np.dot(m, A))
                if L < Lmin:
                    Lmin = L
                    m_best = m
        else:
            for i in range(1, k + 1):
                for index in combinations(range(k), i):
                    m = array([0] * k)
                    m[[index]] = 1
                    L = sqeuclidean(x, dot(m, A))
                    if L < Lmin:
                        Lmin = L
                        m_best = m

        '''
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
        from threading import Thread, local
        from numpy import array, dot, infty, where
        from scipy.spatial.distance import sqeuclidean
        from copy import copy
        from itertools import combinations

        class SearchThread(Thread):
            def __init__(self, params):
                Thread.__init__(self)
                self.x = params.x
                self.A = params.A
                self.m = params.m # the guess (per thread) with 1 "turned on"
            def run(self):
                # generate the combinations as candidate positions
                L_min = sqeuclidean(self.x, dot(self.m, self.A))

                is_active = True
                while is_active:
                    best_candidate = None
                    # choose a best candidate to "turn on" from the rest
                    for i in where(self.m == 0)[0]:
                        candidate_m = copy(self.m)
                        candidate_m[i] = 1 # try turning on 1 cluster
                        candidate_L = sqeuclidean(
                                self.x, dot(candidate_m, self.A))
                        if candidate_L < L_min:
                            # update if better than the current one
                            best_candidate = i
                            L_min = candidate_L
                    if best_candidate:
                        is_active = True
                        self.m[best_candidate] = 1
                    else:
                        # if no better frontier could be found, set inactive
                        is_active = False # break
                return

        n_components = len(m0)
        L0 = sqeuclidean(x, dot(m0, A))
        m_best = m0

        separate_search_threads = list()
        if self.is_disjoint:
            for h in range(n_components): # init each search thread
                m = array([0] * n_components)
                m[h] = 1
                L = sqeuclidean(x, dot(m, A))
                if L < L0:
                    L0 = L
                    m_best = m
        else:
            for h in range(n_components): # init each search thread
                m = array([0] * n_components)
                m[h] = 1
                # create thread local data
                local_data = local()
                local_data.x = x
                local_data.A = A
                local_data.m = m

                s = SearchThread(local_data)
                s.start() # run 
                separate_search_threads.append(s)

            for h in range(n_components): # collect results and select
                separate_search_threads[h].join()
                thread_result_L = sqeuclidean(x,
                        dot(separate_search_threads[h].m, A))
                if thread_result_L < L0:
                    L0 = thread_result_L
                    m_best = separate_search_threads[h].m

        return m_best

