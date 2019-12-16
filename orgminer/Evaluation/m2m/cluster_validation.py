# -*- coding: utf-8 -*-

"""This module provides a set of intrinsic evaluation measures for the 
latent clusters (grouping of resources in the organizational models). 

Notes
-----
To conduct the validation of clusters, the expected inputs of the 
included functions should be both the clusters and the resource feature 
table (profile matrix), rather than an organizational model instance.

Under the situation where the clustering was derived with no feature 
table was used but rather from a graph-like structure, the modularity 
measure should be used. The inputs should be the clusters and the
original graph-like structure from which the clustering was derived.
"""
from warnings import warn

# TODO: is it appropriate using silhouette score for overlapped clusters?
def silhouette_score(clu, X, metric='euclidean'):
    """Calculate the silhouette scores of the latent clustering.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    X : DataFrame
        The corresponding resource feature table used for deriving the 
        resource groups.
    metric : str, optional
        Distance measure used for deriving the resource groups. Refer to
        `scipy.spatial.distance <https://docs.scipy.org/doc/scipy/
        reference/spatial.distance.html>`_.

    Returns
    -------
    scores : dict
        The result silhouette score (of each resource, respectively).

    Notes
    -----
    This is merely a wrap-up of function
    `sklearn.metrics.silhouette_samples
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html>`_.
    """
    from sklearn.metrics import silhouette_samples
    resources = list(X.index)
    labels = [-1] * len(resources)
    for ig, g in enumerate(clu):
        for r in g:
            labels[resources.index(r)] = ig

    scores_samples = silhouette_samples(X.values, labels, metric=metric)
    scores = dict()
    for ir, r in enumerate(resources):
        scores[r] = scores_samples[ir]
    return scores


def variance_explained_score(clu, X):
    """Calculate the degree of variance explained by the latent 
    clustering, i.e. Calinski-Harabasz score.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    X : DataFrame
        The corresponding resource feature table used for deriving the 
        resource groups.

    Returns
    -------
    float
        The result score representing the degree of variance explained.

    Notes
    -----
    This is merely a wrap-up of function
    `sklearn.metrics.calinski_harabasz_score
    <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html>`_.
    """
    from sklearn.metrics import calinski_harabasz_score 
    resources = list(X.index)
    labels = [-1] * len(resources)
    for ig, g in enumerate(clu):
        for r in g:
            labels[resources.index(r)] = ig
    return calinski_harabasz_score(X.values, labels)


def variance_explained_percentage(clu, X):
    """Calculate the percentage of variance explained by the latent 
    clustering.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    X : DataFrame
        The corresponding resource feature table used for deriving the 
        resource groups.

    Returns
    -------
    float
        The result percentage of variance explained (in range 0-100).
    """
    var_between = _variance_between_cluster(clu, X)
    var_within = _variance_within_cluster(clu, X)
    return 100 * var_between / (var_between + var_within)


def _variance_within_cluster(clu, X):
    """Calculate the variance for objects within the same clusters.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    X : DataFrame
        The corresponding resource feature table used for deriving the 
        resource groups.

    Returns
    -------
    var_between : float
        The result within-cluster variance.
    """
    from numpy import mean, sum
    var_within = 0

    for ig, g in enumerate(clu):
        g_mean = mean(X.loc[g].values, axis=0)
        var_within += sum((X.loc[g].values - g_mean) ** 2) # W_k
    var_within /= (len(X) - len(clu)) # N - k
    return var_within


def _variance_between_cluster(clu, X):
    """Calculate the variance for objects between different clusters.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    X : DataFrame
        The corresponding resource feature table used for deriving the 
        resource groups.

    Returns
    -------
    var_between : float
        The result between-cluster variance.
    """
    from numpy import mean, sum
    var_between = 0
    samples_mean = mean(X.values, axis=0)

    for ig, g in enumerate(clu):
        g_mean = mean(X.loc[g].values, axis=0)
        var_between += len(g) * sum((g_mean - samples_mean) ** 2) # B_k
    var_between /= (len(clu) - 1) # k - 1
    return var_between


def modularity(clu, G, weight=None):
    """Calculate the modularity score [1]_ of the latent clustering.
    
    Parameters
    ----------
    clu : list of frozensets
        Resource groups in an organizational model.
    G : NetworkX Graph or DiGraph
        The corresponding resource feature table used for deriving the 
        resource groups.
    weight : str, optional
        The name of the key defining the weight value of edges in `G`.

    Returns
    -------
    q : float
        The result modularity score.

    References
    ----------
    .. [1] "Graph Clustering Methods", Equation (11.39), Sect.
       11.3.3 in book, J. Han, J. Pei, M. Kamber. (2011). *Data Mining:
       Concepts and Techniques*.
    """
    from networkx import is_directed, restricted_view
    q = 0.0
    if is_directed(G):
        """Note: Casting from DiGraph to Graph needs to be configured 
        manually otherwise NetworkX would approach this in an arbitrary 
        fashion.
        """
        from itertools import combinations
        undirected_edge_list = list()
        # consider only pairwise links
        for pair in combinations(G.nodes, r=2):
            if (G.has_edge(pair[0], pair[1]) and 
                G.has_edge(pair[1], pair[0])):
                undirected_edge_wt = 0.5 * (
                    G[pair[0]][pair[1]]['weight'] +
                    G[pair[1]][pair[0]]['weight'])
                undirected_edge_list.append(
                    (pair[0], pair[1], {'weight': undirected_edge_wt}))
            else:
                pass
        G.clear()
        del G
        from networkx import Graph
        G = Graph()
        G.add_edges_from(undirected_edge_list)
        del undirected_edge_list[:]
        warn('DiGraph casted to Graph.', RuntimeWarning)
    else:
        pass

    sum_total_wt = sum([wt for (u, v, wt) in G.edges.data('weight')])
    for g in clu:
        sub_g = restricted_view(G, nodes=list(g), edges=[])
        if weight is None:
            sum_within_cluster_edges = len(list(sub_g.edges)) # l_i
        else:
            sum_within_cluster_edges = sum([wt
                for (u, v, wt) in sub_g.edges.data(weight)]) # l_i
        sum_within_cluster_degree = sum([deg
            for node, deg in sub_g.degree(weight=weight)]) # d_i
        q += (
            (sum_within_cluster_edges / sum_total_wt) - 
            (sum_within_cluster_degree / (2 * sum_total_wt)) ** 2)
    return q

