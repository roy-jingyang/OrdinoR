# -*- coding: utf-8 -*-

# TODO This module needs to be rewritten.

import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

# Unsupervised evaluation requires the resource profile matrix as input
def within_cluster_variance(X, labels, is_overlapped=False):
    total_within_cluster_var = 0
    if is_overlapped:
        label_cluster = defaultdict(lambda: set())
        for i in range(len(labels)):
            for l in labels[i]:
                label_cluster[l].add(i)
        for ul, cluster in label_cluster.items():
            mask = np.array(list(cluster))
            cent = np.mean(X[mask], axis=0) # mean (the medoid)
            dist_to_cent = cdist(
                    X[mask], cent.reshape((1, cent.shape[0]))).ravel()
            if len(X[mask]) < 2: # single element cluster
                within_cluster_var = 0
            else:
                within_cluster_var =  sum(dist_to_cent ** 2) / \
                        (len(X[mask]) - 1)
            total_within_cluster_var += within_cluster_var
    else:
        for ul in np.unique(labels):
            mask = (labels == ul)
            cent = np.mean(X[mask], axis=0)
            dist_to_cent = cdist(
                    X[mask],
                    cent.reshape((1, cent.shape[0])),
                    metric='euclidean')
            total_within_cluster_var += np.var(dist_to_cent, dtype=np.float64)
    return total_within_cluster_var

