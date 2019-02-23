# -*- coding: utf-8 -*-

'''
This module provides a set of intrinsic evaluation measure for the latent
clusters (grouping of resources in the organizational models. For extrinsic
evaluation measure, see ./cluster_comparison.py.

Note that to conduct the validation of clusters, the expected input of these
implemented methods should be the resource feature table (profile matrix) and
the membership labels related to the clusters, rather than an organizational 
model instance.
'''

# TODO: is it appropriate using silhouette score for overlapped clusters?
def silhouette_score(
        clu, X, proximity_metric='euclidean'):
    from numpy import mean, amin
    from scipy.spatial.distance import cdist
    score = list()
    count = 0

    for g in clu:
        for r in g:
            if len(g) == 1:
                # set silhouette score to 0 for size-1 clusters
                score.append(0)
            else:
                r_profile = X.loc[r].values.reshape(1, len(X.loc[r]))
                # a(o)
                avg_intra_dist = mean(cdist(
                    r_profile,
                    X.loc[list(other_r for other_r in g if other_r != r)],
                    metric=proximity_metric))

                # b(o)
                avg_inter_dist = list()
                for other_g in clu:
                    if not other_g == g:
                        avg_inter_dist.append(mean(cdist(
                            r_profile,
                            X.loc[list(other_g)],
                            metric=proximity_metric)))
                if len(avg_inter_dist) == 0:
                    min_avg_inter_dist = 0
                else:
                    min_avg_inter_dist = amin(avg_inter_dist)

                score.append((min_avg_inter_dist - avg_intra_dist) /
                        max(avg_intra_dist, min_avg_inter_dist))

    return mean(score)

