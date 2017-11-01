#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
from EvaluationOptions import Unsupervised

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

    # logrithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    # build resource social network G (with linear tf.) with same indexing
    G = squareform(pdist(profile_mat, metric='euclidean'))
    G = 1 - ((G - G.min()) / (G.max() - G.min()))
    G = nx.from_numpy_matrix(G)
    num_edges_old = len(G.edges)

    # search settings
    threshold_value_MIN = 0.0
    threshold_value_MAX = 1.0
    threshold_value = threshold_value_MIN
    search_range = list()
    scoring_results = list()

    print('Grid search applied, searching range [{}, {}] with step {}'.format(
        threshold_value_MIN, threshold_value_MAX, threshold_value_step))

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
        # only account for valid solution
        if len(np.unique(labels)) > 1:
            # calculating within-cluster variance
            total_within_cluster_var = Unsupervised.within_cluster_variance(
                    profile_mat, labels, is_overlapped=False)

            search_range.append(len(np.unique(labels)))
            #TODO: remove silhouette score
            scoring_results.append((
                threshold_value,
                (total_within_cluster_var),
                labels))
                #Unsupervised.silhouette_score_raw(profile_mat, labels)))
        else:
            pass
        threshold_value += threshold_value_step


    print('Warning: ONLY VALID solutions are accounted.')

    # visualizing the results

    #f, axarr = plt.subplots(1, sharex=True)
    plt.figure(0)
    plt.xlabel('#clusters')
    plt.ylabel('Score - within cluster variance var(#clusters)')
    x = np.array(search_range)
    y = np.array([sr[1] for sr in scoring_results])
    # highlight the original data points
    plt.plot(x, y, 'b*-')
    for i in range(len(x)):
        plt.annotate('[{}]'.format(i), xy=(x[i], y[i]))
    plt.show()

    # select a solution as output
    print('Select the result (by solution_id) under the desired settings:')
    solution_id = int(input())
    solution = scoring_results[solution_id]
    print('Solution [{}] selected:'.format(solution_id))
    print('threshold = {}, '.format(solution[0]))
    print('score: var(k) = {}'.format(solution[1]))
    #TODO
    #print('Silhouette score = {}'.format(solution[3]))

    entities = defaultdict(lambda: set())
    for i in range(len(solution[2])):
        entity_id = int(solution[2][i])
        entities[entity_id].add(resource_index[i])

    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

