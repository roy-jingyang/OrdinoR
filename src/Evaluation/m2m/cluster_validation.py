# -*- coding: utf-8 -*-

'''
This module provides a set of intrinsic evaluation measures for the latent
clusters (grouping of resources in the organizational models. For extrinsic
evaluation measure, see ./cluster_comparison.py.

Note that to conduct the validation of clusters, the expected input of these
implemented methods should be the clusters and the resource feature table 
(profile matrix), rather than an organizational model instance.
'''

# TODO: is it appropriate to define the silhouette score for singletons as 0?
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
    import numpy as np
    from numpy import mean, sum
    var_between = 0
    samples_mean = mean(X.values, axis=0)

    for ig, g in enumerate(clu):
        g_mean = mean(X.loc[g].values, axis=0)
        var_between += len(g) * sum((g_mean - samples_mean) ** 2)    # B_k
    var_between /= len(clu) - 1                                         # k - 1
    return var_between

