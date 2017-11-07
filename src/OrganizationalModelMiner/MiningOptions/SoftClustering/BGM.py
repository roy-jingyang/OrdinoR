#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial.distance import cdist
from EvaluationOptions import Unsupervised

#def mine(cases, threshold_value_step):
def mine(cases):
    print('Applying Variational Bayesian Gaussian Mixture Model:')

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

    '''
    # search settings: threshold for cutting
    threshold_value_MIN = 0.1
    threshold_value_MAX = 1.0
    threshold_value = threshold_value_MIN
    search_range = list()
    scoring_results = list()

    print('Grid search applied, searching range [{}, {}] with step {}'.format(
        threshold_value_MIN, threshold_value_MAX, threshold_value_step))
    '''
    bgm = BayesianGaussianMixture(
            n_components=len(resource_index) - 1,
            covariance_type='spherical',
            n_init=100,
            init_params='random').fit(profile_mat)
            init_params='kmeans').fit(profile_mat)
    resp = bgm.predict_proba(profile_mat)
    determined = resp > 0
    # shape: (n_samples,)
    labels = list()
    for row in determined:
        labels.append(tuple(l for l in np.nonzero(row)[0]))
    labels = np.array(labels).ravel()
    is_overlapped = labels.dtype != np.int64
    total_within_cluster_var = Unsupervised.within_cluster_variance(
            profile_mat, labels, is_overlapped)
    solution = ((bgm.score(profile_mat), total_within_cluster_var),
            labels)

    '''
    # Use BGM for probabilistic cluster assignments
    while threshold_value < threshold_value_MAX:
        #TODO
        bgm = BayesianGaussianMixture(
                n_components=len(resource_index),
                covariance_type='spherical',
                n_init=20,
                init_params='random').fit(profile_mat)

        if threshold_value >= 1:
            print('Warning: Disjoint clustering option unavailable!')
            exit()
            # shape: (n_samples,)
            #labels = bgm.predict(profile_mat) 
        else:
            resp = bgm.predict_proba(profile_mat)
            determined = resp >= threshold_value
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
            search_range.append(threshold_value)
            scoring_results.append((
                threshold_value,
                (bgm.score(profile_mat),
                    total_within_cluster_var),
                labels,
                is_overlapped))

        threshold_value += threshold_value_step

    print('Warning: ONLY VALID solutions are accounted.')

    # visualizing the results

    #plt.figure(0)
    f, axarr = plt.subplots(2, sharex=True)
    plt.xlabel('threshold')
    #plt.ylabel('Score - likelihood')
    axarr[0].set_ylabel('Score - likelihood')
    axarr[1].set_ylabel('Score - within cluster variance var(#clusters)')
    x = np.array(search_range)
    #y = np.array([sr[1] for sr in scoring_results])
    y0 = np.array([sr[1][0] for sr in scoring_results])
    y1 = np.array([sr[1][1] for sr in scoring_results])
    # highlight the original data points
    #plt.plot(x, y, 'b*-')
    axarr[0].plot(x, y0, 'b*-')
    axarr[1].plot(x, y1, 'b*-')
    for i in range(len(x)):
        #plt.annotate('[{}]'.format(i), xy=(x[i], y[i]))
        axarr[0].annotate('[{}]'.format(i), xy=(x[i], y0[i]))
        axarr[1].annotate('[{}]'.format(i), xy=(x[i], y1[i]))
        if scoring_results[i][3] == True: # is overlapped
            #plt.plot(x[i], y[i], marker='o', markeredgecolor='r')
            axarr[0].plot(x[i], y0[i], marker='o', markeredgecolor='r')
            axarr[1].plot(x[i], y1[i], marker='o', markeredgecolor='r')
    plt.show()

    # select a solution as output
    print('Select the result (by solution_id) under the desired settings:')
    solution_id = int(input())
    solution = scoring_results[solution_id]
    print('Solution [{}] selected:'.format(solution_id))
    print('threshold = {}, '.format(solution[0]))
    #print('score: likelihood = {}'.format(solution[1]))
    print('score: likelihood = {}, var(k) = {}'.format(
        solution[1][0], solution[1][1]))

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
    '''
    print('score: likelihood = {}, var(k) = {}'.format(
        solution[0][0], solution[0][1]))

    entities = defaultdict(lambda: set())
    for i in range(len(solution[1])):
        if is_overlapped:
            for l in solution[1][i]:
                entity_id = l
                entities[l].add(resource_index[i])
        else:
            # only 1 cluster assigned for each
            entity_id = int(solution[1][i])
            entities[entity_id].add(resource_index[i])

    print('Clustering results are ' + \
            ('overlapped.' if is_overlapped == True else 'disjoint.'))
    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

