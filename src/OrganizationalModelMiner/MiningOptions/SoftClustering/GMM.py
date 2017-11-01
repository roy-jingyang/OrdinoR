#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from EvaluationOptions import Unsupervised

def mine(cases, threshold):
    print('Applying Gaussian Mixture Model:')
    if threshold < 0 or threshold >= 1:
        print('Threshold for converting to determined clustering must value' +\
                ' between [0, 1) !')
        print('Warning: Disjoint clustering option unavailable!')
        exit(1)

    # constructing the performer-activity matrix from event logs
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = []

    for caseid, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            if activity in activity_index:
                pass
            else:
                activity_index.append(activity)
            counting_mat[resource][activity] += 1

    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count

    # logrithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    # search settings: k_cluster
    k_cluster_MIN = 1
    k_cluster_MAX = len(resource_index)
    k_cluster_step = 1
    k_cluster = k_cluster_MIN
    search_range = list()
    scoring_results = list()
    # search settings: threshold_value

    # Use GMM for probabilistic cluster assignments
    while k_cluster <= k_cluster_MAX:
        gmm = GaussianMixture(
            n_components=k_cluster,
            init_params='random').fit(profile_mat)

        if threshold == 1:
            print('Warning: Disjoint clustering option unavailable!')
            exit()
            # shape: (n_samples,)
            #labels = gmm.predict(profile_mat) 
        else:
            #TODO
            resp = gmm.predict_proba(profile_mat)
            determined = resp >= threshold
            # shape: (n_samples,)
            labels = list()
            for row in determined:
                labels.append(tuple(l for l in np.nonzero(row)[0]))
            labels = np.array(labels).ravel()

        # evaluation (unsupervised) goes here
        # only account for valid solution
        is_overlapped = labels.dtype != np.int64
        if is_overlapped or len(np.unique(labels)) > 1:
            total_within_cluster_var = Unsupervised.within_cluster_variance(
                    profile_mat, labels, is_overlapped)
            search_range.append(k_cluster)
            scoring_results.append((
                k_cluster,
                (gmm.score(profile_mat),
                    total_within_cluster_var,
                    gmm.bic(profile_mat)),
                labels,
                is_overlapped))

        k_cluster += k_cluster_step

    print('Warning: ONLY VALID solutions are accounted.')

    # visualizing the results

    #plt.figure(0)
    f, axarr = plt.subplots(3, sharex=True)
    plt.xlabel('#clusters')
    #plt.ylabel('Score - likelihood')
    axarr[0].set_ylabel('Score - likelihood')
    axarr[1].set_ylabel('Score - within cluster variance var(#clusters)')
    axarr[2].set_ylabel('Score - BIC')
    x = np.array(search_range)
    #y = np.array([sr[1] for sr in scoring_results])
    y0 = np.array([sr[1][0] for sr in scoring_results])
    y1 = np.array([sr[1][1] for sr in scoring_results])
    y1 = np.array([sr[1][2] for sr in scoring_results])
    # highlight the original data points
    #plt.plot(x, y, 'b*-')
    axarr[0].plot(x, y0, 'b*-')
    axarr[1].plot(x, y1, 'b*-')
    axarr[2].plot(x, y1, 'b*-')
    for i in range(len(x)):
        #plt.annotate('[{}]'.format(i), xy=(x[i], y[i]))
        axarr[0].annotate('[{}]'.format(i), xy=(x[i], y0[i]))
        axarr[1].annotate('[{}]'.format(i), xy=(x[i], y1[i]))
        axarr[2].annotate('[{}]'.format(i), xy=(x[i], y1[i]))
        if scoring_results[i][3] == True: # is overlapped
            #plt.plot(x[i], y[i], marker='o', markeredgecolor='r')
            axarr[0].plot(x[i], y0[i], marker='o', markeredgecolor='r')
            axarr[1].plot(x[i], y1[i], marker='o', markeredgecolor='r')
            axarr[2].plot(x[i], y1[i], marker='o', markeredgecolor='r')
    plt.show()

    # select a solution as output
    print('Select the result (by solution_id) under the desired settings:')
    solution_id = int(input())
    solution = scoring_results[solution_id]
    print('Solution [{}] selected:'.format(solution_id))
    print('#clusters = {}, '.format(solution[0]))
    #print('score: likelihood = {}'.format(solution[1]))
    print('score: likelihood = {}, var(k) = {}, BIC = {}'.format(
        solution[1][0], solution[1][1], solution[1][2]))

    entities = defaultdict(lambda: set())
    for i in range(len(solution[2])):
        if solution[3] == True: # overlapped
            for l in solution[2][i]:
                entity_id = l
                entities[l].add(resource_index[i])
        else:
            # only 1 cluster assigned for each
            entity_id = int(solution[2][i])
            entities[entity_id].add(resource_index[i])

    print('Clustering results are ' + \
            ('overlapped.' if solution[3] == True else 'disjoint.'))
    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

