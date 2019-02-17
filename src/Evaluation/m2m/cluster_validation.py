# -*- coding: utf-8 -*-

'''
Ref: Aggarwal, C. C. (2015). Data Mining: The Textbook. Springer Publishing
Company, Incorporated.

This module provides a set of intrinsic evaluation measure for the latent
clusters (grouping of resources in the organizational models. For extrinsic
evaluation measure, see ./cluster_comparison.py.

Note that to conduct the validation of clusters, the expected input of these
implemented methods should be the resource feature table (profile matrix) and
the membership labels related to the clusters, rather than an organizational 
model instance.
'''

# TODO: is it approapriate using silhouette score for overlapped clusters?
def silhouette_score(
        clu, X, proximity_metric='euclidean'):
    from numpy import mean, amin
    from scipy.spatial.distance import cdist
    score = list()
    count = 0

    for g in clu:
        for r in g:
            if len(g) == 1:
                score.append(0)
            else:
                other_members = list(x for x in g if x is not r)
                avg_intra_dist = mean(cdist(
                    X.loc[r].values.reshape(1, len(X.loc[r])),
                    X.loc[other_members],
                    metric=proximity_metric))

                avg_inter_dist = list()
                for other_g in clu:
                    if r not in other_g:
                        avg_inter_dist.append(mean(cdist(
                            X.loc[r].values.reshape(1, len(X.loc[r])),
                            X.loc[list(other_g)],
                            metric=proximity_metric)))
                min_inter_dist = amin(avg_inter_dist)

                score.append((min_inter_dist - avg_intra_dist) /
                        max(avg_intra_dist, min_inter_dist))

    return mean(score)

