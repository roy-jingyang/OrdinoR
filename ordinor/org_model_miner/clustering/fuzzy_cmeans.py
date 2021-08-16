"""
Fuzzy C-Means for overlapping clustering
"""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean as dist
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans

from ordinor.org_model_miner.community import mja
from ordinor.org_model_miner._helpers import cross_validation_score
import ordinor.exceptions as exc

from .agglomerative_hierarchical import ahc

class FCM:
    """
    Fuzzy C-Means clustering [1]_ implemented based on `SciKit-Fuzzy
    <http://pythonhosted.org/scikit-fuzzy/>`_.

    Methods
    -------
    fit_predict(X) : Fit and then predict labels for the input samples.

    References
    ----------
    .. [1] Tan, P. N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).
       *Introduction to Data Mining*.
    """

    def __init__(self,
                 n_components, 
                 tol=1e-6, 
                 p=2, 
                 n_init=1, 
                 max_iter=1000,
                 means_init=None):
        """
        Instantiate an FCM class instance.

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
        """
        Initialize a list of random guesses of fuzzy pseudo partition w.

        Parameters
        ----------
        n_init : int
            Number of iterations to be used.

        Returns
        -------
        l_rand_w : list
            The result of random initialization.
        """
        l_rand_w = list()
        for init in range(n_init):
            # random init, constraint: row sum = 1.0
            w = list()
            for i in range(shape[0]):
                weights = np.randint(1, 10, shape[1]) 
                weights = weights / sum(weights)
                w.append(weights)
            l_rand_w.append(np.array(w))
        return l_rand_w


    def fit_predict(self, X):
        """
        Fit and then predict labels of the input samples.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        best_w : array-like, shape (n_samples, n_components)
            The weight values.
        """

        best_w = None
        if self._means_init is not None:
            # if seed centroids given, compute initial fuzzy pseudo partition
            w = np.zeros((len(X), self._n_components))
            exp = 1 / (self._p - 1)
            for i in range(len(X)):
                l_sqd_xi_c = [
                    np.power(dist(X[i,:], self._means_init[q,:]), 2)
                    for q in range(self._n_components)
                ]
                if 0.0 in l_sqd_xi_c:
                    # special case: current point is one of the centroids
                    cntr_cluster_ix = l_sqd_xi_c.index(0.0)
                    w[i,cntr_cluster_ix] = 1.0 # leave others 0
                else:
                    for j in range(self._n_components):
                        sqd_xi_cj = np.power(
                            dist(X[i,:], self._means_init[j,:]), 
                            2
                        )
                        w[i,j] = (
                            np.power((1 / sqd_xi_cj), exp)
                            / 
                            np.sum([
                                np.power(
                                    (1 / np.power(dist(X[i,:], self._means_init[q,:]), 2)), 
                                    exp
                                ) 
                                for q in range(self._n_components)
                            ])
                        )
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


def _fcm(profiles, n_groups, threshold, init='random', n_init=100): 
    print('Applying overlapping clustering-based FCM:')
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
    elif init == 'random':
        warm_start = False
    else:
        raise exc.InvalidParameterError(
            param='init',
            reason='Can only be one of {"mja", "ahc", "kmeans", "plain", "random"}'
        )

    # step 1. Train the model
    if warm_start:
        init_means = list()
        for g in init_groups:
            init_means.append(np.mean(profiles.loc[list(g)].values, axis=0))
        fcm_model = FCM(
            n_components=n_groups,
            n_init=1,
            means_init=np.array(init_means))
    else:
        fcm_model = FCM(
            n_components=n_groups,
            n_init=n_init)

    # step 2. Derive the clusters as the end result
    fpp = fcm_model.fit_predict(profiles.values)

    groups = defaultdict(set)
    for i, belonging_factor in enumerate(fpp):
        if threshold is None:
            #threshold = median(belonging_factor[belonging_factor != 0])
            threshold = 1.0 / n_groups
        membership = np.array([p >= threshold for p in belonging_factor])

        # check if any valid membership exists for the resource based on
        # the selection of the threshold
        if membership.any():
            for j in np.nonzero(membership)[0]:
                groups[j].add(profiles.index[i])
        else: # invalid, have to choose the maximum one or missing the resource
            groups[np.argmax(belonging_factor)].add(profiles.index[i])

    return [frozenset(g) for g in groups.values()]


def fcm(profiles, 
        n_groups, 
        threshold, 
        init='random', 
        n_init=100,
        search_only=False): 
    """
    Apply Fuzzy C-Means (FCM) [1]_ to discover resource groups.

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
    threshold : float
        A given threshold value in range [0, 1.0] for producing
        determined clustering from the fuzzy clustering results from FCM.
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
    .. [1] Tan, P. N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).
       *Introduction to Data Mining*.
    """
    if type(n_groups) is int:
        return _fcm(profiles, n_groups, threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _fcm(profiles, n_groups[0], threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
            score = cross_validation_score(
                X=profiles, miner=_fcm,
                miner_params={
                    'n_groups': k,
                    'threshold': threshold,
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
            return _fcm(profiles, best_k, threshold, init, n_init)
    else:
        raise exc.InvalidParameterError(
            param='n_groups',
            reason='Expected an int or a non-empty list'
        )
