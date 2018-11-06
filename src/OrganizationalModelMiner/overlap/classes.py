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
    
    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        best_m : array-like, shape (n_samples, n_components), component membership
        '''

        from numpy import array, dot, infty, zeros, absolute
        from numpy.random import randint, choice
        from numpy.linalg import pinv

        M_init_values_visited = set() # pool for storing used seeds
        best_score = None
        best_M = None
        init = 0
        while init < self.n_init:
            # Do initialization based on selected setting
            if self.M_init is not None:
                M = self.M_init
            else:
                # Initialize a random guess of membership M
                # Random initialization of a binary matrix
                while True:
                    M = zeros((len(X), self.n_components))
                    if self.is_disjoint:
                        for row in M:
                            row[randint(self.n_components)] = 1 # only 1
                    else:
                        for row in M:
                            row[choice(range(self.n_components),
                                size=randint(1, self.n_components + 1),
                                replace=False)] = 1

                    M.flags.writeable = False # set RO, entries immutable
                    bM = M.data.tobytes()
                    if bM not in M_init_values_visited:
                        # if seed unused before
                        M_init_values_visited.add(bM)
                        break

            # Start fitting with valid initialized value of M
            iteration = -1
            current_log_likelihood = None
            delta_log_likelihood = infty
            converged = False

            #print('Fitting MOC model-{}:'.format(init + 1), end=' ')
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

                if absolute(delta_log_likelihood) < self.tol:
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
                    break

            if is_valid:
                #print('Final score =\t{:.8f}'.format(current_log_likelihood))
                if best_score is None or current_log_likelihood > best_score:
                    best_score = current_log_likelihood
                    best_M = M.copy()
                init += 1
            else:
                print('(Result invalid. Re-run the fitting.)')
                pass
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

    def fit_predict(self, X):
        '''Fit and then predict labels for samples.

        Parameters
        __________
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        w : array-like, shape (n_samples, n_components), weights
        '''

        from numpy import array, infty, zeros, sum, power, dot, amax, absolute
        from numpy.random import randint, choice
        from scipy.spatial.distance import euclidean as dist

        best_sse = None
        best_w = None
        init = 0
        while init < self.n_init:
            # Select an initial fuzzy pseudo-partition, i.e. the values for weights
            if self.means_init is not None:
                # if initial centroids provided, compute initial fuzzy partition
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
            else:
                # random init, constraint: row sum = 1.0
                w = list()
                for i in range(len(X)):
                    weights = randint(1, 10, self.n_components) 
                    weights = weights / sum(weights)
                    w.append(weights)
                w = array(w)

            # Start fitting with valid initialized value of w
            iteration = -1
            current_sse = None
            delta_sse = infty
            converged = False

            #print('Fitting FCM model-{}:'.format(init + 1), end=' ')
            while not converged:
                iteration += 1
                if iteration >= self.max_iter:
                    print('[Warning] Model did not converged within a max. of '
                            + '{} iterations (delta= {:.4f}).'.format(
                                self.max_iter, self.tol))
                    return w

                #print('\n\tIteration {}:'.format(iteration), end=' ')
                prev_sse = current_sse

                # Compute the centroid for each cluster using the pseudo-partition
                # (eq. 9.2)
                cntr = dot(power(w.T, self.p), X) # shape = (n_comp, n_features)
                for j in range(self.n_components):
                    cntr[j,:] /= sum(
                        [power(w[i,j], self.p) for i in range(len(X))])

                # Recompute the fuzzy psuedo-partition (eq. 9.3)
                exp = 1 / (self.p - 1)
                for i in range(len(X)):
                    for j in range(self.n_components):
                        sqd_xi_cj = power(dist(X[i,:], cntr[j,:]), 2)
                        w[i,j] = (
                                power((1 / sqd_xi_cj), exp)
                                /
                                sum([power(
                                    (1 / power(dist(X[i,:], cntr[q,:]), 2)), 
                                    exp) for q in range(self.n_components)])
                                )

                # Calculate the new SSE:
                current_sse = self.error(X, cntr, w)

                if prev_sse is not None:
                    delta_sse = current_sse - prev_sse

                # until: change in SSE is below certain threshold
                if absolute(delta_sse) < self.tol:
                    converged = True
                    if delta_sse <= 0:
                        pass
                    else:
                        # current solution is worse (minimizing)
                        #print('\tDELTA > 0: Pick the last solution instead.')
                        current_sse = prev_sse
                        w = prev_w.copy()
                    #print('\nModel converged')

            # check if the solution is valid
            is_valid = True
            for j in range(self.n_components):
                if not w[:,j].any():
                    is_valid = False
                    break

            if is_valid:
                #print('Final SSE =\t{:.8f}'.format(current_sse))
                if best_sse is None or current_sse < best_sse:
                    best_sse = current_sse
                    best_w = w.copy()
                init += 1
            else:
                print('(Result invalid. Re-run the fitting.)')
                pass

        return best_w
    
    # (eq. 9.1)
    def error(self, X, centroids, weights):
        from numpy import dot, power
        from scipy.spatial.distance import euclidean as dist

        total_sse = 0.0
        for j in range(self.n_components):
            for i in range(len(X)):
                sqd_xi_cj = power(dist(X[i,:], centroids[j,:]), 2)
                total_sse += power(weights[i,j], self.p) * sqd_xi_cj
        return total_sse

