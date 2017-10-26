#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import spline
from collections import defaultdict
from sklearn.metrics import silhouette_score

#TODO
def threshold(cases, threshold_value_step):
    print('Applying Metrics based on Joint Activities:')
    # constructing the performer-activity matrix from event logs
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = list()
    # adjacency-list for performer-activity
    for caseid, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            if activity not in activity_index:
                activity_index.append(activity)
            counting_mat[resource][activity] += 1

    G = nx.Graph()
    # adjacency-matrix for performer-activity
    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count

    # build resource social network G (with scaling) with same indexing
    G = squareform(pdist(profile_mat, metric='euclidean'))
    G = (G - G.min()) / (G.max() - G.min())
    G = nx.from_numpy_matrix(G)
    num_edges_old = len(G.edges)

    # search settings
    threshold_value_MIN = 0.0
    threshold_value_MAX = 1.0
    threshold_value = threshold_value_MIN
    search_range = list()
    scoring_results = defaultdict(lambda: list())

    while threshold_value <= threshold_value_MAX:
        # perform graph cutting for clustering:
        graph = G.copy()

        # iterate through all edges
        edges_to_remove = list()
        for u, v, wt in graph.edges.data('weight'):
            # remove edge if weight lower than threshold
            if wt < threshold_value:
                edges_to_remove.append((u, v))
        graph.remove_edges_from(edges_to_remove)
        print('{:.2%} edges'.format(len(edges_to_remove) / num_edges_old) +
                ' have been filtered by threshold {}.'.format(threshold_value)) 

        labels = np.empty((len(G), 1))
        # obtain the connected components as discovered results
        cluster_id = -1 # consecutive numbers as entity id
        for comp in nx.connected_components(graph):
            cluster_id += 1
            for u in list(comp):
                labels[list(G.nodes).index(u)] = cluster_id
        labels = labels.ravel()

        # evaluation (unsupervised) goes here
        score = np.nan if len(np.unique(labels)) < 2 else silhouette_score(
                profile_mat, labels.ravel(), metric='euclidean')
        scoring_results['silhouette_score'].append(
                (threshold_value, score, labels))

        search_range.append(threshold_value)
        threshold_value += threshold_value_step


    # visualizing the results

    plt.figure(0)
    plt.title('threshold value of MJA -- Score')
    plt.xlabel('threshold value')
    plt.ylabel('Score')
    y = np.array([sr[1] for sr in scoring_results['silhouette_score']])
    # filter out the invalid (nan) scores
    index = ~np.isnan(y)
    y = y[index]
    x = np.array(search_range)[index]
    # smoothing the line
    x_new = np.linspace(min(x), max(x), 1000 * len(x))
    y_smooth = spline(x, y, x_new)
    plt.plot(x_new, y_smooth)
    # highlight the original data points
    plt.plot(x, y, 'ro')
    for i in range(len(x)):
        plt.annotate(str(x[i]), xy=(x[i], y[i]))
    plt.show()

    # select a solution as output
    print('Select the result under the desired settings:')
    threshold_value = float(input())

    entities = defaultdict(lambda: set())
    for result in scoring_results['silhouette_score']: 
        if threshold_value == result[0]:
            for i in range(len(result[2])):
                entity_id = int(result[2][i])
                entities[entity_id].add(resource_index[i])

    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

