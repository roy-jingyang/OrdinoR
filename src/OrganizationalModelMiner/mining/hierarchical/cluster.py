#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of hierarchical organizational mining
methods, based on the use of clustering techniques. These methods are "profile-
based", meaning that resource profiles should be used as input.
Methods include:
    1. Agglomerative Hierarchical Clustering (Song & van der Aalst)
'''

def ahc(profiles,
        n_groups,
        method='single',
        metric='euclidean'):
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
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
    Returns:
        og_hcy: DataFrame
            The hierarchical structure as a pandas DataFrame, with resource ids
            as indices, levels of the hierarchy as columns, and group ids as
            the values, e.g. for 20 resources placed in a 5-level hierarhical
            structure with 8 groups at the lowest level, there should be 20
            rows and 5 columns in the DataFrame, and the values should be in
            range of 0 to 7.
    '''

    from scipy.cluster import hierarchy
    Z = hierarchy.linkage(profiles, method=method, metric=metric)
    # the hierachical tree as a matrix where each column corresponds to a
    # specific level
    mx_tree = hierarchy.cut_tree(Z, n_clusters=range(1, n_groups + 1))
    # wrap as DataFrame og_hcy
    from pandas import DataFrame
    return DataFrame(mx_tree, index=profiles.index)

