"""
Model-based Overlapping Clustering
"""

from itertools import combinations
from functools import partial
from operator import itemgetter
import multiprocessing
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import sqeuclidean
from sklearn.cluster import KMeans
from pandas import DataFrame

from ordinor.org_model_miner.community import mja
from ordinor.org_model_miner._helpers import cross_validation_score
import ordinor.exceptions as exc

from .agglomerative_hierarchical import ahc

class MOC:
    """
    Model-based Overlapping Clustering (MOC) [1]_ implemented with
    multiprocessing.

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
                 n_components, 
                 tol=1e-6, 
                 n_init=1, 
                 max_iter=1000, 
                 M_init=None,
                 is_disjoint=False):
        """
        Instantiate an MOC class instance.

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
            A Boolean flag indicating whether a disjoint result is 
            required.
        """
        self._n_components = n_components
        self._tol = tol
        self._n_init = n_init if M_init is None else 1
        self._max_iter = max_iter
        self._M_init = M_init
        self._is_disjoint = is_disjoint


    def _init_M(self, n_init, shape):
        """
        Initialize a list of random guesses of membership M.

        Parameters
        ----------
        n_init : int
            Number of iterations to be used.

        Returns
        -------
        l_rand_M : list
            The result of random initialization.
        """
        l_rand_M = list()
        visited = set() # pool for storing used seeds
        for init in range(n_init):
            # Random initialization (non-repeat)
            while True:
                M = np.zeros(shape)
                if self._is_disjoint:
                    for row in M:
                        row[np.randint(self._n_components)] = 1 # only 1
                else:
                    for row in M:
                        row[np.choice(
                            range(shape[1]),
                            size=np.randint(1, shape[1] + 1),
                            replace=False)
                        ] = 1

                M.flags.writeable = False # set RO, entries immutable
                bM = M.data.tobytes()
                if bM not in visited:
                    # if seed unused before
                    visited.add(bM)
                    l_rand_M.append(M)
                    break
        return l_rand_M
    

    def fit_predict(self, X):
        """
        Fit and then predict labels of the input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        best_m : array-like, shape (n_samples, n_components)
            The membership component in the model.
        """
        if self._M_init is not None:
            l_M = [self._M_init.copy()]
        else:
            l_M = self._init_M(self._n_init, shape=(len(X), self._n_components))
            
        def _e_m(M):
            # Start fitting with valid initialized value of M
            iteration = -1
            current_log_likelihood = None
            delta_log_likelihood = np.infty
            converged = False

            while not converged:
                iteration += 1
                if iteration >= self._max_iter:
                    exc.warn_runtime(
                        f'Not yet converged within {self._max_iter} iterations'
                    )
                    return M

                #print('\n\tIteration {}:'.format(iteration), end=' ')
                prev_log_likelihood = current_log_likelihood

                # Alternate between updating M and A
                # update A: assume loss function squared Euclidean Distance
                pinv_M = np.linalg.pinv(M) 
                A = np.dot(pinv_M, X)

                # update M: for each row, apply appropriate search algorithms
                new_M = list()
                for i in range(X.shape[0]):
                    #m_best = self._enumerate(X[i,:], M[i,:], A)
                    m_best = self._dynamicm(X[i,:], M[i,:], A)
                    new_M.append(m_best)
                prev_M = M.copy()
                M = np.array(new_M)

                # Calculate new log-likelihood:
                current_log_likelihood = self.score(X, M, A)
                #print('score = {:.8f}'.format(current_log_likelihood), end='')

                if prev_log_likelihood is not None: # if not the initial run
                    delta_log_likelihood = (
                        current_log_likelihood - prev_log_likelihood
                    )

                if (delta_log_likelihood is not np.infty 
                    and delta_log_likelihood < self._tol):
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
        """
        Calculate the likelihood score of a solution from fitting the 
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
        #print('Calculating the score: ', end='')
        # calculate alpha_ih
        score_alpha = 0.0
        for h in range(M.shape[1]): # n_components
            M_h = M[:,h]
            pie_h = (1.0 / X.shape[0]) * len(M_h[M_h == True])
            for i in range(M.shape[0]): # n_samples
                # Bernoulli
                alpha_ih = (
                    np.power(pie_h, M[i,h])
                    * 
                    np.power((1 - pie_h), (1 - M[i,h]))
                )
                score_alpha += np.log(alpha_ih)

        # calculate the Bregman divergence
        # currently Squared Euclidean Distance
        score_divergence = 0.0
        MA = np.dot(M, A)
        for i in range(X.shape[0]): # n_components
            for j in range(X.shape[1]): # n_features
                score_divergence += sqeuclidean(X[i,j], MA[i,j])

        score = score_alpha - score_divergence
        #print('{:.4f}'.format(score))
        return score
    

    '''
    # search over all possible settings in the naive way
    def _enumerate(self, x, m0, A):
        L0 = sqeuclidean(x, np.dot(m0, A))
        Lmin = L0
        k = len(m0)
        m_best = m0

        # Exhaustively enumerate all possible settings sum_{i = 1 to k} C^i_k
        # start from 1 ensures that at least one scalar with value True
        for i in range(1, k + 1):
            for index in combinations(range(k), i):
                m = np.array([0] * k)
                m[[index]] = 1
                L = sqeuclidean(x, np.dot(m, A))
                if L < Lmin:
                    Lmin = L
                    m_best = m
        return m_best
    '''


    def _search_thread(self, x, A, m):
        # generate the combinations as candidate positions
        L_min = sqeuclidean(x, np.dot(m, A))

        is_active = True
        while is_active: # search each "level"
            best_candidate = None
            # choose a best candidate to "turn on" from the rest
            for i in np.where(m == 0)[0]:
                candidate_m = m.copy()
                candidate_m[i] = 1 # try turning on 1 cluster
                candidate_L = sqeuclidean(
                    x, np.dot(candidate_m, A))
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
        return m, sqeuclidean(x, np.dot(m, A))


    # search on each separate threads with different initial setting
    # greedily proceed on the fly (idea similar to DP)
    def _dynamicm(self, x, m0, A):
        """
        A dynamic algorithm (using parallel threads) to search for a 
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
        n_components = len(m0)
        L0 = sqeuclidean(x, np.dot(m0, A))
        m_best = m0

        if self._is_disjoint:
            for h in range(n_components):
                m = np.array([0] * n_components)
                m[h] = 1
                L = sqeuclidean(x, np.dot(m, A))
                if L < L0:
                    L0 = L
                    m_best = m
        else:
            separate_search_threads = list()
            for h in range(n_components): # init each search thread
                m = np.array([0] * n_components)
                m[h] = 1
                separate_search_threads.append(m)

            with multiprocessing.Pool() as p:
                m_best = min(
                    p.map(
                        partial(self._search_thread, x, A),
                        separate_search_threads
                    ), 
                    key=itemgetter(1))[0]
        return m_best


def _moc(profiles, n_groups, init='random', n_init=100):
    print('Applying overlapping clustering-based MOC:')
    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        if init == 'mja':
            init_groups = mja(profiles, n_groups)  
        elif init == 'ahc':
            init_groups, _ = ahc(profiles, n_groups)
        elif init == 'kmeans':
            init_groups = list(set() for i in range(n_groups))
            labels = KMeans(n_clusters=n_groups, random_state=0).fit_predict(
                profiles)
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[labels[i]].add(r)
        else: # init == 'plain'
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        print('Initialization done using {}:'.format(init))

        m = DataFrame(np.zeros((len(profiles), n_groups)), index=profiles.index)
        for i, g in enumerate(init_groups):
            for r in g:
                m.loc[r][i] = 1 # set the membership matrix as init input
    elif init == 'random':
        warm_start = False
    else:
        raise exc.InvalidParameterError(
            param='init',
            reason='Can only be one of {"mja", "ahc", "kmeans", "plain", "random"}'
        )

    # step 1. Train the model
    if warm_start:
        moc_model = MOC(n_components=n_groups, M_init=m.values, n_init=1)
    else:
        moc_model = MOC(n_components=n_groups, n_init=n_init)

    # step 2. Derive the clusters as the end result
    mat_membership = moc_model.fit_predict(profiles.values)

    groups = defaultdict(set)
    for i in range(len(mat_membership)):
        # check if any valid membership exists for the resource based on
        # the results predicted by the obtained MOC model
        if mat_membership[i,:].any(): # valid if at least belongs to 1 group
            for j in np.nonzero(mat_membership[i,:])[0]:
                groups[j].add(profiles.index[i])
        else: # invalid (unexpected exit)
            print(mat_membership)
            raise exc.AlgorithmRuntimeError(
                reason='Could not produce a valid clustering',
                suggestion='Try changing the initialization?'
            )

    return [frozenset(g) for g in groups.values()]


def moc(profiles, n_groups, init='random', n_init=100,
    search_only=False):
    """Apply the Model-based Overlapping Clustering (MOC) to discover 
    resource groups [1]_.

    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : pandas.DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.
    search_only : bool, optional, default False
        A Boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    See Also
    --------
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    if type(n_groups) is int:
        return _moc(profiles, n_groups, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _moc(profiles, n_groups[0], init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
            # NOTE: use Cross Validation
            score = cross_validation_score(
                X=profiles, miner=_moc,
                miner_params={
                    'n_groups': k,
                    'init': init,
                    'n_init': n_init
                },
                proximity_metric='euclidean'
            )

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _moc(profiles, best_k, init, n_init)
    else:
        raise exc.InvalidParameterError(
            param='n_groups',
            reason='Expected an int or a non-empty list'
        )
