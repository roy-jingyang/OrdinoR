# -*- coding: utf-8 -*-

'''
This module provides a set of intrinsic evaluation measures for the latent
clusters (grouping of resources in the organizational models. For extrinsic
evaluation measure, see ./cluster_comparison.py.

Note that to conduct the validation of clusters, the expected input of these
implemented methods should be the clusters and the resource feature table 
(profile matrix), rather than an organizational model instance;

Despite the situation where the clustering is derived with no feature table
is used but rather from a graph-like structure, where the modularity of
clustering should be used. The inputs should include the clusters and the
original graph-like structure from which the clustering is derived.

'''

# TODO: is it appropriate using silhouette score for overlapped clusters?
def silhouette_score(
        clu, X, metric='euclidean'):
    '''
    This is merely a wrap-up of the function:
        sklearn.metrics.silhouette_samples
    '''

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
    '''
    from numpy import mean, amin
    from scipy.spatial.distance import cdist
    scores = dict()
    count = 0

    for g in clu:
        for r in g:
            if len(g) < 2:
                # set silhouette score to 0 for size-1 clusters
                # set to value 0 ("unclear", should be 'nan')
                #scores[r] = 'nan'
                scores[r] = 0
            else:
                r_profile = X.loc[r].values.reshape(1, len(X.loc[r]))
                # a(o)
                avg_intra_dist = mean(cdist(
                    r_profile,
                    X.loc[list(other_r for other_r in g if other_r != r)],
                    metric=metric))

                # b(o)
                avg_inter_dist = list()
                for other_g in clu:
                    if not other_g == g:
                        avg_inter_dist.append(mean(cdist(
                            r_profile,
                            X.loc[list(other_g)],
                            metric=metric)))
                if len(avg_inter_dist) == 0:
                    min_avg_inter_dist = 0
                else:
                    min_avg_inter_dist = amin(avg_inter_dist)

                score = ((min_avg_inter_dist - avg_intra_dist) /
                    max(avg_intra_dist, min_avg_inter_dist))
                scores[r] = score                  # silhouette(o)

    return scores
    '''

def variance_explained_score(
    clu, X):
    '''
    This is merely a wrap-up of the function:
        sklearn.metrics.calinski_harabasz_score
    which is equal to
        variance_between_cluster / variance_within_cluster
    '''
    
    from sklearn.metrics import calinski_harabasz_score 
    resources = list(X.index)
    labels = [-1] * len(resources)
    for ig, g in enumerate(clu):
        for r in g:
            labels[resources.index(r)] = ig
    return calinski_harabasz_score(X.values, labels)

def variance_explained_percentage(
    clu, X):
    '''
    This is another version which uses percentage to characterize:
        variance_between_cluster / (var_between... + var_within...)
    '''
    var_between = _variance_between_cluster(clu, X)
    var_within = _variance_within_cluster(clu, X)

    return 100 * var_between / (var_between + var_within)

def _variance_within_cluster(
    clu, X):
    from numpy import mean, sum
    var_within = 0

    for ig, g in enumerate(clu):
        g_mean = mean(X.loc[g].values, axis=0)
        var_within += sum((X.loc[g].values - g_mean) ** 2)       # W_k
    var_within /= len(X) - len(clu)                                 # N - k
    return var_within


def _variance_between_cluster(
    clu, X):
    from numpy import mean, sum
    var_between = 0
    samples_mean = mean(X.values, axis=0)

    for ig, g in enumerate(clu):
        g_mean = mean(X.loc[g].values, axis=0)
        var_between += len(g) * sum((g_mean - samples_mean) ** 2)    # B_k
    var_between /= len(clu) - 1                                         # k - 1
    return var_between

def modularity(
    clu, G, weight=None):
    '''
    Reference:
        "Graph Clustering Methods",
        Equation (11.39), Sect. 11.3.3, Data Mining: Concepts and Techniques,
        J. Han, J. Pei, M. Kamber
    '''
    from networkx import is_directed, restricted_view
    q = 0.0
    if is_directed(G):
        # Casting from DiGraph to Graph needs to be configured manually
        # otherwise NetworkX would approach this in an arbitrary fashion
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
        print('[Warning] DiGraph casted to Graph.')
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

