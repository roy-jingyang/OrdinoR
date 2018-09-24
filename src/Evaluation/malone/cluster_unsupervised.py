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

def extended_modularity(resources, org_model, sn_model):
    resources = sorted(list(resources))
    N = len(resources)

    labels = defaultdict(lambda: set())
    for entity_id, entity in org_model.items():
        entity = list(entity)
        for res in entity:
            labels[res].add(entity_id)

    print('<k> = {:.3f}'.format(
        np.mean([deg for (u, deg) in sn_model.degree()])))
    m = sum([w for (u, v, w) in sn_model.edges.data('weight')])
    modularity = 0.0

    # Shen
    for entity_id, entity in org_model.items():
        entity = list(entity)
        for i in range(len(entity) - 1): # for any two different points
            u = entity[i]
            for j in range(i + 1, len(entity)):
                v = entity[j]
                Ou = len(labels[u])
                Ov = len(labels[v])
                A = 1 if sn_model.has_edge(u, v) or sn_model.has_edge(v, u) \
                        else 0

                ku = sn_model.degree(u, weight='weight')
                kv = sn_model.degree(v, weight='weight')
                modularity += (1.0 / (Ou * Ov)) * \
                        (A - ku * kv / (2 * m))

    modularity *= 1 / (2 * m)

    return modularity

