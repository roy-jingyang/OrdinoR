# -*- coding: utf-8 -*-

"""This module contains the implementation of hierarchical organizational 
mining methods, based on the use of clustering techniques. These methods
are "profile-based", meaning that resource profiles should be used as
input.
"""
def _ahc(profiles, n_groups, method='single', metric='euclidean'):
    """The basic Agglomerative Hierarchical Clustering algorithm [1]_.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    method : str, optional
        Choice of methods for merging clusters at each iteration.
        Defaults to 'single'.
    metric : str, optional
        Choice of metrics for measuring the distance while calculating 
        distance.
        Defaults to ``euclidean``, meaning that euclidean distance is 
        used for measuring distance.

    Returns
    -------
    list of frozensets
        Discovered organizational groups.
    og_hcy : DataFrame
        The hierarchical structure (dendrogram) as a pandas DataFrame,
        with resource ids as indices, levels of the hierarchy as columns,
        and group ids as the values. 
        E.g. for 20 resources placed in a 5-level hierarchical structure
        with 8 groups at the lowest level, there should be 20 rows and 5
        columns in the DataFrame, and the values should be in range of 0
        to 7.
    
    See Also
    --------
    scipy.cluster.hierarchy

    References
    ----------
    .. [1] Song, M., & van der Aalst, W. M. P. (2008). Towards
    comprehensive support for organizational mining. Decision Support
    Systems, 46(1), 300-317.
    """
    print('Applying hierarchical organizational mining using AHC:')
    from scipy.cluster import hierarchy
    Z = hierarchy.linkage(profiles, method=method, metric=metric)
    # the hierachical tree as a matrix where each column corresponds to a
    # specific level
    mx_tree = hierarchy.cut_tree(Z, n_clusters=range(1, n_groups + 1))
    # wrap as DataFrame og_hcy
    from pandas import DataFrame
    og_hcy = DataFrame(mx_tree, index=profiles.index)

    from collections import defaultdict
    groups = defaultdict(lambda: set())
    # add by each resource
    for i in range(len(og_hcy.index)):
        groups[og_hcy.iloc[i,-1]].add(og_hcy.index[i])

    #print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()], og_hcy


def ahc(profiles, n_groups, method='single', metric='euclidean',
    search_only=False):
    """A wrapped method for ``_ahc``.

    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    method : str, optional
        Choice of methods for merging clusters at each iteration.
        Defaults to 'single'.
    metric : str, optional
        Choice of metrics for measuring the distance while calculating 
        distance.
        Defaults to ``euclidean``, meaning that euclidean distance is 
        used for measuring distance.
    search_only: bool, optional
        A boolean flag indicating whether to search for the number of
        groups only or to perform cluster analysis based on the search
        result. 
        Defaults to False, i.e., to perform cluster analysis after 
        searching.

    Returns
    -------
    best_ogs : list of frozensets
        Discovered organizational groups.
    best_og_hcy : DataFrame
        The hierarchical structure (dendrogram) as a pandas DataFrame,
        with resource ids as indices, levels of the hierarchy as columns,
        and group ids as the values. 
        E.g. for 20 resources placed in a 5-level hierarchical structure
        with 8 groups at the lowest level, there should be 20 rows and 5
        columns in the DataFrame, and the values should be in range of 0
        to 7.

    See Also
    --------
    _ahc
    """
    if len(n_groups) == 1:
        return _ahc(profiles, n_groups[0], method, metric)
    else:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import cross_validation_score
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

