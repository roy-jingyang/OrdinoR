# -*- coding: utf-8 -*-

"""This module contains definitions of several classes that are used by 
the clustering-based methods for organizational model discovery.

See Also
--------
orgminer.OrganizationalModelMiner.clustering.overlap
"""
from warnings import warn

class MOC:
    """This class implements the method of Model-based Overlapping 
    Clustering (MOC) [1]_.

    Methods
    -------
    fit_predict(X)
        Fit and then predict labels for the input samples.
    score(X, M, A)
        Calculate the likelihood score of a solution from fitting the 
        samples

    Notes
    -----
    In the current implementation, Gaussian distributions are assumed for
    the latent categories of the data samples, thus squared Euclidean
    distance measure is used as the Bregman divergence.

    References
    ----------
    .. [1] Banerjee, A., Krumpelman, C., Ghosh, J., Basu, S., & Mooney,
       R. J. (2005). Model-based overlapping clustering. In
       *Proceedings of the eleventh ACM SIGKDD International Conference 
       on Knowledge Discovery in Data mining*, pp. 532-537. ACM.
       `<https://doi.org/10.1145/1081870.1081932>`_
    """

    def __init__(self, 
        n_components, tol=1e-6, n_init=1, max_iter=1000, M_init=None,
        is_disjoint=False):
        """Instantiate an MOC class instance.

        Parameters
        ----------
        n_components : int
            Number of components in the model.
        tol : float, optional, default 1e-6
            The convergence threshold.
        n_init : int, optional, default 1
            Number of initializations to perform. The best results 
            are kept. This parameter would be override `M_init` if 
            specified.
        max_iter : int, optional, default 1000
            Number of iterative alternating updates to run.
        M_init : array-like, shape (n_samples, n_components), optional, 
        default None
            User-provided initial membership matrix M (binary-valued). 
            If ``None``, random initialization is used.
        is_disjoint : bool, optional, default False
            A boolean flag indicating whether a disjoint result is 
            required.
        """
        self._n_components = n_components
        self._tol = tol
        self._n_init = n_init if M_init is None else 1
        self._max_iter = max_iter
        self._M_init = M_init
        self._is_disjoint = is_disjoint


    def _init_M(self, n_init, shape):
        """Initialize a list of random guesses of membership M.

        Parameters
        ----------
        n_init : int
            Number of iterations to be used.

        Returns
        -------
        l_rand_M : list
            The result of random initialization.
        """
        from numpy import zeros
        from numpy.random import randint, choice

        l_rand_M = list()
        visited = set() # pool for storing used seeds
        for init in range(n_init):
            # Random initialization (non-repeat)
            while True:
                M = zeros(shape)
                if self._is_disjoint:
                    for row in M:
                        row[randint(self._n_components)] = 1 # only 1
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
        """Fit and then predict labels for the input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        best_m : array-like, shape (n_samples, n_components)
            The membership component in the model.
        """
        from numpy import array, dot, infty
        from numpy.linalg import pinv

        if self._M_init is not None:
            l_M = [self._M_init.copy()]
        else:
            l_M = self._init_M(self._n_init, shape=(len(X), self._n_components))
            
        def _e_m(M):
            # Start fitting with valid initialized value of M
            iteration = -1
            current_log_likelihood = None
            delta_log_likelihood = infty
            converged = False

            while not converged:
                iteration += 1
                if iteration >= self._max_iter:
                    warn('Convergence not reached within {} iterations'.format(
                        self._max_iter), RuntimeWarning)
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
                    #m_best = self._enumerate(X[i,:], M[i,:], A)
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

                if delta_log_likelihood is not infty and delta_log_likelihood < self._tol:
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
            for j in range(self._n_components):
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
        """Calculate the likelihood score of a solution from fitting the 
        samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        M : array-like, shape (n_samples, n_components)
            The membership component in the model.
        A : array-like, shape (n_components, n_features)
            The parameter component in the model.
        
        Returns
        -------
        float
            The result score.

        See Also
        --------
        fit_predict
        """
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
    

    '''
    # search over all possible settings in the naive way
    def _enumerate(self, x, m0, A):
        from numpy import array, dot
        from scipy.spatial.distance import sqeuclidean
        from itertools import combinations

        L0 = sqeuclidean(x, dot(m0, A))
        Lmin = L0
        k = len(m0)
        m_best = m0

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
        return m_best
    '''


    def _search_thread(self, x, A, m):
        from numpy import array, dot, infty, where
        from scipy.spatial.distance import sqeuclidean
        # generate the combinations as candidate positions
        L_min = sqeuclidean(x, dot(m, A))

        is_active = True
        while is_active: # search each "level"
            best_candidate = None
            # choose a best candidate to "turn on" from the rest
            for i in where(m == 0)[0]:
                candidate_m = m.copy()
                candidate_m[i] = 1 # try turning on 1 cluster
                candidate_L = sqeuclidean(
                    x, dot(candidate_m, A))
                if candidate_L < L_min:
                    # update if better than the current one
                    best_candidate = i
                    L_min = candidate_L
            if best_candidate is not None:
                is_active = True
                m[best_candidate] = 1
            else:
                # if no better frontier could be found, set inactive
                is_active = False # break
        return m, sqeuclidean(x, dot(m, A))


    # search on each separate threads with different initial setting
    # greedily proceed on the fly (idea similar to DP)
    def _dynamicm(self, x, m0, A):
        """A dynamic algorithm (using parallel threads) to search for a 
        "best" solution.

        Parameters
        ----------
        x : array-like, shape (n_features, )
            A row of the input samples.
        m0 : array-like, shape (n_components, )
            A row of the membership component in the model.
        A : array-like, shape (n_components, n_features)
            The parameter component in the model.
        
        Returns
        -------
        m_best : array-like, shape (n_components, )
            A "best" solution with respect to the given values.

        Notes
        -----
        The idea is similar to Dynamic Programming. The returned result 
        is prone to be sub-optimal. Refer to the paper for more 
        information.

        See Also
        --------
        fit_and_predict
        """
        from numpy import array, dot, infty, where
        from scipy.spatial.distance import sqeuclidean

        n_components = len(m0)
        L0 = sqeuclidean(x, dot(m0, A))
        m_best = m0

        if self._is_disjoint:
            for h in range(n_components):
                m = array([0] * n_components)
                m[h] = 1
                L = sqeuclidean(x, dot(m, A))
                if L < L0:
                    L0 = L
                    m_best = m
        else:
            from functools import partial
            from multiprocessing import Pool
            from operator import itemgetter

            separate_search_threads = list()
            for h in range(n_components): # init each search thread
                m = array([0] * n_components)
                m[h] = 1
                separate_search_threads.append(m)

            with Pool() as p:
                m_best = min(
                    p.map(
                        partial(self._search_thread, x, A),
                        separate_search_threads
                    ), 
                    key=itemgetter(1))[0]
        return m_best


class FCM:
    """This class implements the method of Fuzzy C-Means clustering [2]_.

    The implementation is based on `SciKit-Fuzzy 
    <http://pythonhosted.org/scikit-fuzzy/>`_.

    Methods
    -------
    fit_predict(X) : Fit and then predict labels for the input samples.

    References
    ----------
    .. [2] Tan, P. N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).
       *Introduction to Data Mining*.
    """

    def __init__(self,
        n_components, tol=1e-6, p=2, n_init=1, max_iter=1000,
        means_init=None):
        """Instantiate an FCM class instance.

        Parameters
        ----------
        n_components : int
            Number of clusters expected.
        tol : float, optional, default 1e-6
            The convergence threshold.
        p : float, optional, default 2
            The exponentiation value. When p = 1, fuzzy c-means reduces
            to K-means algorithm; when p is set to larger values, the 
            partitioning becomes fuzzier (approaching global centroid).
        n_init : int, optional, default 1
            Number of initializations to perform. The best results 
            are kept. This parameter would be override `means_init if 
            specified.
        max_iter : int, optional, default 1000
            Number of iterative alternating updates to run.
        means_init : array-like, shape (n_components, n_features), 
        optional, default None
            User-provided initial guess of centroids. If ``None``, random
            initialization is used and assigns random-valued weights for
            samples.
        """
        self._n_components = n_components
        self._tol = tol
        self._p = p
        self._n_init = n_init if means_init is None else 1
        self._max_iter = max_iter
        self._means_init = means_init


    def _init_w(self, n_init, shape):
        """Initialize a list of random guesses of fuzzy pseudo partition 
        w.

        Parameters
        ----------
        n_init : int
            Number of iterations to be used.

        Returns
        -------
        l_rand_w : list
            The result of random initialization.
        """
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
        """Fit and then predict labels for the input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        best_w : array-like, shape (n_samples, n_components)
            The weight values.
        """
        from numpy import array, infty, zeros, sum, power, dot, amax
        from scipy.spatial.distance import euclidean as dist
        from skfuzzy.cluster import cmeans

        best_w = None
        if self._means_init is not None:
            # if seed centroids given, compute initial fuzzy pseudo partition
            w = zeros((len(X), self._n_components))
            exp = 1 / (self._p - 1)
            for i in range(len(X)):
                l_sqd_xi_c = [power(dist(X[i,:], self._means_init[q,:]), 2)
                        for q in range(self._n_components)]
                if 0.0 in l_sqd_xi_c:
                    # special case: current point is one of the centroids
                    cntr_cluster_ix = l_sqd_xi_c.index(0.0)
                    w[i,cntr_cluster_ix] = 1.0 # leave others 0
                else:
                    for j in range(self._n_components):
                        sqd_xi_cj = power(dist(X[i,:], self._means_init[j,:]), 2)
                        w[i,j] = (
                                power((1 / sqd_xi_cj), exp)
                                / sum([power(
                                    (1 / power(dist(X[i,:], self._means_init[q,:]), 2)), 
                                    exp) for q in range(self._n_components)]))
            l_w = [w.copy()]
        else:
            l_w = self._init_w(self._n_init, shape=(len(X), self._n_components))

        def _e_m(w):
            _, w, w0, _, sse, _, _ = cmeans(data=X.T,
                    c=self._n_components, m=self._p, error=self._tol, 
                    maxiter=self._max_iter, init=w.T)
             
            # check if the solution is valid
            is_valid = True
            for j in range(self._n_components):
                if not w[:,j].any():
                    is_valid = False
                    break

            if is_valid:
                #print('Final SSE =\t{:.8f}'.format(sse[-1]))
                return w.copy().T, sse[-1]
        
        l_fitted_w = list(map(_e_m, l_w))
        best_w, best_sse = max(l_fitted_w, key=lambda x: x[1])
        return best_w

