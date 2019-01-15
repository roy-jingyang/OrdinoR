# -*- coding: utf-8 -*-

'''
This module contains the implementation of hierarchical organizational mining
methods, based on the use of clustering techniques. These methods are "profile-
based", meaning that resource profiles should be used as input.
Methods include:
    1. Agglomerative Hierarchical Clustering (Song & van der Aalst)
'''

def _ahc(
        profiles, n_groups,
        method='single', metric='euclidean'):
    '''
    This method implements the basic agglomerative hierarchical clustering
    algorithm in clustering analysis, proposed by Song & van der Aalst.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        method: str, optional
            Choice of methods for calculating the newly formed cluster and 
            each sample point. The default is 'single'.Refer to
            scipy.cluster.hierarchy.linkage for more detailed explanation.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
    Returns:
        list of frozensets
            A list of organizational groups.
        og_hcy: DataFrame
            The hierarchical structure as a pandas DataFrame, with resource ids
            as indices, levels of the hierarchy as columns, and group ids as
            the values, e.g. for 20 resources placed in a 5-level hierarhical
            structure with 8 groups at the lowest level, there should be 20
            rows and 5 columns in the DataFrame, and the values should be in
            range of 0 to 7.
    '''

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

def ahc(
        profiles, n_groups,
        method='single', metric='euclidean',
        search_only=False):
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: iterable
            The (range of) number of groups to be discovered.
        method: str, optional
            Choice of methods for calculating the newly formed cluster and 
            each sample point. The default is 'single'.Refer to
            scipy.cluster.hierarchy.linkage for more detailed explanation.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        search_only: boolean, optional
            Determine whether to search for the number of groups only or to
            perform cluster analysis based on the search result. The default is
            to perform cluster analysis after searching.
    Returns:
        best_ogs: list of frozensets
            A list of organizational groups.
        best_og_hcy: DataFrame
            The hierarchical structure as a pandas DataFrame, with resource ids
            as indices, levels of the hierarchy as columns, and group ids as
            the values, e.g. for 20 resources placed in a 5-level hierarhical
            structure with 8 groups at the lowest level, there should be 20
            rows and 5 columns in the DataFrame, and the values should be in
            range of 0 to 7.
    '''
    if len(n_groups) == 1:
        return _ahc(profiles, n_groups[0], method, metric)
    else:
        from OrganizationalModelMiner.utilities import cross_validation_score
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
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

