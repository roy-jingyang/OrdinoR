#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.spatial.distance import cdist
from collections import defaultdict

# Unsupervised evaluation requires the description matrix as input

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

def silhouette_score(X, labels, is_overlapped=False):
    if not is_overlapped:
        if len(np.unique(labels)) > X.shape[0] - 1:
            return np.nan
        else:
            return metrics.silhouette_score(X, labels, metric='euclidean')
    else:
        #TODO
        exit()

def calinski_harabaz_score(X, labels, is_overlapped=False):
    if not is_overlapped:
        return metrics.calinski_harabaz_score(X, labels)
    else:
        #TODO
        exit()

def extended_modularity(resources, org_model, sn_model):
    resources = list(resources)
    N = len(resources)

    labels = defaultdict(lambda: set())
    mat_cluster = np.eye(N) # diag set to 1
    for entity_id, entity in org_model.items():
        # build the labels (for Oi, Oj)
        for res in entity:
            labels[res].add(entity_id)

        # build the similarity matrix (for the delta function)
        # avoid repeatedly counting the simultaneous appearance
        for i in range(len(entity) - 1):
            u = resources.index(entity[i])
            for j in range(i + 1, len(entity)):
                v = resources.index(entity[j])
                # symmetric
                mat_cluster[u][v] = 1
                mat_cluster[v][u] = 1

    m = len(sn_model.edges)
    modularity = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N): # for any two different items
            res_i = resources.index(i)
            res_j = resources.index(j)

            delta = mat_cluster[i][j] # delta(C^i, C^j)
            Oi = len(labels[res_i])
            Oj = len(labels[res_j])
            Aij = sn_model[res_i][res_j]['weight'] \
                    if sn_model.has_edge(res_i, res_j) else 0
            deg_i = sn_model.degree(res_i)
            deg_j = sn_model.degree(res_j)

            modularity += 1 / (Oi * Oj) * (
                    (Aij - deg_i * deg_j / (2 * m)) * delta)

    modularity *= 1 / (2 * m)

    return modularity

