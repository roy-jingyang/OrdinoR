# -*- coding: utf-8 -*-

# TODO: in the current implementation, we assume a Gaussian distribution for
# the latent categories of the data samples, thus the Bregman divergence is the
# square Euclidean distance.
class MOC:
    '''
    This class implements the method of Model-based Overlapping Clustering.

    n_components: int, defaults to 1.
        The number of components in the model.

    tol: float, defaults to 1e-6.
        The convergence threshold.

    n_init: int, defaults to 1.
        The number of initializations to perform. The best results are kept.
        This parameter would be override if M_init is present.

    max_iter: int, defaults to 1000.
        The number of iterative alternating updates to run.

    M_init: array-like, shape (n_samples, n_components), optional
        The user-provided initial membership matrix M (binary), defaults to None.
        If None, random initialization is used.

    is_disjoint: Boolean, defaults to False
        The Boolean flag indicating whether a disjoint result (the reduced
        case) is required.

    '''

    def __init__(self, 
            n_components=1, tol=1e-6, n_init=1, max_iter=1000, M_init=None,
            is_disjoint=False):
        self.n_components = n_components
        self.tol = tol
        self.n_init = n_init if M_init is None else 1
        self.max_iter = max_iter
        self.M_init = M_init
        self.is_disjoint = is_disjoint

    def _init_M(self, n_init, shape):
        '''Initialize a list of random guesses of membership M.

        Parameters
        __________
        n_init : int

        Returns
        -------
        l_rand_M : list
        '''
        from numpy import zeros
        from numpy.random import randint, choice

        l_rand_M = list()
        visited = set() # pool for storing used seeds
        for init in range(n_init):
            # Random initialization (non-repeat)
            while True:
                M = zeros(shape)
                if self.is_disjoint:
                    for row in M:
                        row[randint(self.n_components)] = 1 # only 1
                else:
                    for row in M:
                        row[choice(range(shape[1]),
                            size=randint(1, shape[1] + 1),
                            replace=False)] = 1

                M.flags.writeable = False # set RO, entries immutable
                bM = M.data.tobytes()
                if bM not in visited:
                    # if seed unused before
                    visited.add(bM)
                    l_rand_M.append(M)
                    break
        return l_rand_M
    
    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        best_m : array-like, shape (n_samples, n_components), component membership
        '''
        from numpy import array, dot, infty
        from numpy.linalg import pinv

        if self.M_init is not None:
            l_M = [self.M_init.copy()]
        else:
            l_M = self._init_M(self.n_init, shape=(len(X), self.n_components))
            
        def _e_m(M):
            # Start fitting with valid initialized value of M
            iteration = -1
            current_log_likelihood = None
            delta_log_likelihood = infty
            converged = False

            while not converged:
                iteration += 1
                if iteration >= self.max_iter:
                    print('[Warning] Model did not converged within a max. of '
                            + '{} iterations (delta= {:.4f}).'.format(
                                self.max_iter, self.tol))
                    return M

                #print('\n\tIteration {}:'.format(iteration), end=' ')
                prev_log_likelihood = current_log_likelihood

                # Alternate between updating M and A
                # update A: assume loss function squared Euclidean Distance
                pinv_M = pinv(M) 
                A = dot(pinv_M, X)

                # update M: for each row, apply appropriate search algorithms
                new_M = list()
                for i in range(X.shape[0]):
                    m_best = self._dynamicm(X[i,:], M[i,:], A)
                    new_M.append(m_best)
                prev_M = M.copy()
                M = array(new_M)

                # Calculate new log-likelihood:
                current_log_likelihood = self.score(X, M, A)
                #print('score = {:.8f}'.format(current_log_likelihood), end='')

                if prev_log_likelihood is not None: # if not the initial run
                    delta_log_likelihood = (current_log_likelihood
                            - prev_log_likelihood)

                if delta_log_likelihood is not infty and delta_log_likelihood < self.tol:
                    converged = True
                    if delta_log_likelihood >= 0:
                        pass
                    else:
                        # current solution is worse (maximizing), use the last one
                        #print('\tDELTA < 0: Pick the last solution instead.')
                        current_log_likelihood = prev_log_likelihood
                        M = prev_M.copy()
                    #print('\nModel converged with ', end='')

            # check if the solution is valid
            is_valid = True
            for j in range(self.n_components):
                if not M[:,j].any(): # invalid, an empty cluster found
                    is_valid = False
                    return M.copy(), float('-nan') # return as NaN score
            if is_valid:
                #print('Final score =\t{:.8f}'.format(current_log_likelihood))
                return M.copy(), current_log_likelihood
       
        l_fitted_M = list(map(_e_m, l_M))
        best_M, best_score = max(l_fitted_M, key=lambda x: x[1])
        return best_M

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

    # search on each separate threads with different initial setting
    # greedily proceed on the fly (idea similar to DP)
    def _dynamicm(self, x, m0, A):
        from threading import Thread, local
        from numpy import array, dot, infty, where
        from scipy.spatial.distance import sqeuclidean
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
                while is_active: # search each "level"
                    best_candidate = None
                    # choose a best candidate to "turn on" from the rest
                    for i in where(self.m == 0)[0]:
                        candidate_m = self.m.copy()
                        candidate_m[i] = 1 # try turning on 1 cluster
                        candidate_L = sqeuclidean(
                                self.x, dot(candidate_m, self.A))
                        if candidate_L < L_min:
                            # update if better than the current one
                            best_candidate = i
                            L_min = candidate_L
                    if best_candidate is not None:
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
            for h in range(n_components):
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
                # create thread local data copies
                local_data = local()
                local_data.x = x.copy()
                local_data.A = A.copy()
                local_data.m = m.copy()

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


class FCM:
    '''
    Class variables:

    n_components: int, defaults to 1.
        The number of clusters expected to be found.

    tol: float, defaults to 1e-6.
        The convergence threshold.

    p: float, defaults to 2.0.
        The exponentiation value for updating equations.
        When p = 1, fuzzy c-means reduces to traditional K-means algorithm;
        When p gets larger, the partitions grow fuzzier (approaching global
        centroid).

    n_init: int, defaults to 1.
        The number of initializations to perform. The best results are kept.
        This parameter would be override if means_init is present.

    max_iter: int, defaults to 1000.
        The number of iterative alternating updates to run.

    means_init: array-like, shape (n_components, n_features), optional
        The user-provided initial guess of centroids.
        If None, random initialization is used and assigns random-valued 
        weights for samples.
    '''
    def __init__(self,
            n_components=1, tol=1e-6, p=2, n_init=1, max_iter=1000,
            means_init=None):
        self.n_components = n_components
        self.tol = tol
        self.p = p
        self.n_init = n_init if means_init is None else 1
        self.max_iter = max_iter
        self.means_init = means_init

    def _init_w(self, n_init, shape):
        '''Initialize a list of random guesses of fuzzy pseudo partition w.

        Parameters
        __________
        n_init : int

        Returns
        -------
        l_rand_w : list
        '''
        from numpy import array
        from numpy.random import randint, choice

        l_rand_w = list()
        for init in range(n_init):
            # random init, constraint: row sum = 1.0
            w = list()
            for i in range(shape[0]):
                weights = randint(1, 10, shape[1]) 
                weights = weights / sum(weights)
                w.append(weights)
            l_rand_w.append(array(w))
        return l_rand_w

    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        w : array-like, shape (n_samples, n_components), weights
        '''

        from numpy import array, infty, zeros, sum, power, dot, amax
        from scipy.spatial.distance import euclidean as dist
        from skfuzzy.cluster import cmeans

        best_sse = None
        best_w = None
        if self.means_init is not None:
            # if seed centroids given, compute initial fuzzy pseudo partition
            w = zeros((len(X), self.n_components))
            exp = 1 / (self.p - 1)
            for i in range(len(X)):
                l_sqd_xi_c = [power(dist(X[i,:], self.means_init[q,:]), 2)
                        for q in range(self.n_components)]
                if 0.0 in l_sqd_xi_c:
                    # special case: current point is one of the centroids
                    cntr_cluster_ix = l_sqd_xi_c.index(0.0)
                    w[i,cntr_cluster_ix] = 1.0 # leave others 0
                else:
                    for j in range(self.n_components):
                        sqd_xi_cj = power(dist(X[i,:], self.means_init[j,:]), 2)
                        w[i,j] = (
                                power((1 / sqd_xi_cj), exp)
                                / sum([power(
                                    (1 / power(dist(X[i,:], self.means_init[q,:]), 2)), 
                                    exp) for q in range(self.n_components)]))
            l_w = [w.copy()]
        else:
            l_w = self._init_w(self.n_init, shape=(len(X), self.n_components))

        def _e_m(w):
            _, w, w0, _, sse, _, _ = cmeans(data=X.T,
                    c=self.n_components, m=self.p, error=self.tol, 
                    maxiter=self.max_iter, init=w.T)
             
            # check if the solution is valid
            is_valid = True
            for j in range(self.n_components):
                if not w[:,j].any():
                    is_valid = False
                    break

            if is_valid:
                #print('Final SSE =\t{:.8f}'.format(sse[-1]))
                return w.copy().T, sse[-1]
        
        l_fitted_w = list(map(_e_m, l_w))
        best_w, best_sse = max(l_fitted_w, key=lambda x: x[1])
        return best_w

