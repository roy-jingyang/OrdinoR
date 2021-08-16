"""
Gaussian Mixture Models for overlapping clustering
"""

from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from ordinor.org_model_miner.community import mja
from ordinor.org_model_miner._helpers import cross_validation_score
import ordinor.exceptions as exc

from .agglomerative_hierarchical import ahc

def _gmm(profiles, n_groups, threshold, init='random', n_init=100): 
    print('Applying overlapping clustering-based GMM:')
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
        gmm_model = GaussianMixture(
            n_components=n_groups,
            covariance_type='tied',
            tol=1e-9,
            max_iter=1000,
            n_init=1,
            random_state=0,
            means_init=init_means).fit(profiles.values)
    else:
        gmm_model = GaussianMixture(
            n_components=n_groups,
            covariance_type='tied',
            tol=1e-9,
            max_iter=1000,
            n_init=n_init,
            init_params='random').fit(profiles.values)

    # step 2. Derive the clusters as the end result
    posterior_pr = gmm_model.predict_proba(profiles.values)
    '''
    threshold_ub = 0
    for pr in sorted(unique(posterior_pr), reverse=True):
        mbr_mat = posterior_pr >= pr
        if count_nonzero(mbr_mat.any(axis=1)) < len(profiles):
            pass
        else:
            if count_nonzero(mbr_mat.any(axis=0)) == n_groups:
                threshold_ub = pr
                break

    if threshold_ub != 0.0:
        threshold = amin([percentile(posterior_pr, 75), threshold_ub])
        membership_total = posterior_pr >= threshold
    else:
        membership_total = posterior_pr > 0.0
    '''
    membership_total = posterior_pr >= threshold

    groups = defaultdict(set)
    for i, membership in enumerate(membership_total):
        for j in np.nonzero(membership)[0]:
            groups[j].add(profiles.index[i])

    return [frozenset(g) for g in groups.values()]


def gmm(profiles, 
        n_groups, 
        threshold, 
        init='random', 
        n_init=100,
        search_only=False): 
    """
    Apply Gaussian Mixture Model (GMM) to discover resource groups [1]_.

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
        determined clustering from the fuzzy clustering results from GMM.
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
        after search searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    See Also
    --------
    sklearn.mixture.GaussianMixture
    sklearn.cluster.KMeans
    pandas.DataFrame

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    if type(n_groups) is int:
        return _gmm(profiles, n_groups, threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _gmm(profiles, n_groups[0], threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
            # NOTE: use Cross Validation
            score = cross_validation_score(
                X=profiles, miner=_gmm,
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
            return _gmm(profiles, best_k, threshold, init, n_init)
    else:
        raise exc.InvalidParameterError(
            param='n_groups',
            reason='Expected an int or a non-empty list'
        )
