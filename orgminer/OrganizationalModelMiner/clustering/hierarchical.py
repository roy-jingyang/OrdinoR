# -*- coding: utf-8 -*-

"""This module contains the implementation of hierarchical organizational 
mining methods, based on the use of clustering techniques.
"""
def _ahc(profiles, n_groups, method='single', metric='euclidean'):
    """Apply the classic Agglomerative Hierarchical Clustering (AHC) 
    [1]_ to discover resource groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    method : str, optional, default 'single'
        Choice of methods for merging clusters at each iteration.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    list of frozensets
        Discovered resource groups.
    
    See Also
    --------
    scipy.spatial.distance
    sklearn.cluster.AgglomerativeClustering
    pandas.DataFrame

    References
    ----------
    .. [1] Song, M., & van der Aalst, W. M. P. (2008). Towards
       comprehensive support for organizational mining. *Decision Support
       Systems*, 46(1), 300-317.
       `<https://doi.org/10.1016/j.dss.2008.07.002>`_
    """
    print('Applying hierarchical clustering-based AHC:')

    '''
    from scipy.cluster import hierarchy
    Z = hierarchy.linkage(profiles, method=method, metric=metric)

    # the hierachical tree as a matrix where each column corresponds to a
    # specific level
    mat_tree = hierarchy.cut_tree(Z, n_clusters=range(1, n_groups + 1))
    # wrap as DataFrame og_hcy
    from pandas import DataFrame
    og_hcy = DataFrame(mat_tree, index=profiles.index)

    from collections import defaultdict
    groups = defaultdict(set)
    # add by each resource
    for i in range(len(og_hcy.index)):
        groups[og_hcy.iloc[i,-1]].add(og_hcy.index[i])

    return [frozenset(g) for g in groups.values()], og_hcy
    '''
    from sklearn.cluster import AgglomerativeClustering
    if method == 'ward':
        # Ward's method can only work with Euclidean distance
        clustering = AgglomerativeClustering(n_groups,
            affinity='euclidean', linkage=method).fit_predict(profiles)
    else:
        from scipy.spatial.distance import pdist, squareform
        mat_dist = squareform(pdist(profiles, metric=metric))
        clustering = AgglomerativeClustering(n_groups,
            affinity='precomputed', linkage=method).fit_predict(mat_dist)

    from collections import defaultdict
    groups = defaultdict(set)
    # add by each resource
    for i, cluster in enumerate(clustering):
        groups[cluster].add(profiles.index[i])

    return [frozenset(g) for g in groups.values()]


def ahc(profiles, n_groups, method='single', metric='euclidean',
    search_only=False):
    """Apply the classic Agglomerative Hierarchical Clustering (AHC) 
    [1]_ to discover resource groups.
    
    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    method : str, optional, default 'single'
        Choice of methods for merging clusters at each iteration.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
        groups only or to perform cluster analysis based on the search
        result. Defaults to ``False``, i.e., to perform cluster analysis
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` 
        is True).
    best_ogs : list of frozensets
        Discovered resource groups (if `search_only` is False).

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    scipy.spatial.distance
    sklearn.cluster.AgglomerativeClustering
    pandas.DataFrame

    References
    ----------
    .. [1] Song, M., & van der Aalst, W. M. P. (2008). Towards
       comprehensive support for organizational mining. *Decision Support
       Systems*, 46(1), 300-317.
       `<https://doi.org/10.1016/j.dss.2008.07.002>`_
    """
    if type(n_groups) is int:
        return _ahc(profiles, n_groups, method, metric)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _ahc(profiles, n_groups[0], method, metric)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import \
            cross_validation_score
        from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
        from numpy import mean, amax
        for k in n_groups:
            # NOTE: use Cross Validation
            score = cross_validation_score(
                X=profiles, miner=_ahc,
                miner_params={
                    'n_groups': k,
                    'method': method,
                    'metric': metric 
                },
                proximity_metric = metric
            )

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _ahc(profiles, best_k, method, metric)
    else:
        raise TypeError('Invalid type for parameter `{}`: {}'.format(
            'n_groups', type(n_groups)))

