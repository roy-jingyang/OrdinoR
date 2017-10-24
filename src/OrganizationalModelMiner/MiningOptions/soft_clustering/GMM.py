#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture
import copy

def mine(cases, k_clusters, threshold):
    print('Applying Gaussian Mixture Model:')
    # Again, constructing the performer-activity matrix from event logs
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

    # Use GMM for probabilistic cluster assignments
    gmm = GaussianMixture(
        n_components=k_clusters,
        init_params='random').fit(profile_mat)

    if threshold < 0 or threshold > 1:
        print('Threshold for converting to determined clustering must value' +\
                ' between [0, 1]')
        exit(1)
    elif threshold == 1:
        print('Warning: threshold value == 1 results in disjoint clustering.')
        labels = gmm.predict(profile_mat) # shape: (n_samples,)
    else:
        resp = gmm.predict_proba(profile_mat)
        determined = resp >= threshold
        # shape: (n_samples, k_clusters)
        labels = np.array([np.nonzero(row)[0] for row in determined])

    entities = defaultdict(lambda: set())
    for i in range(len(resource_index)):
        resource = resource_index[i]
        if len(np.shape(labels[i])) == 0: # shape: (n_samples,)
            entities[labels[i]].add(resource)
        else: # shape: (n_samples, k_clusters)
            for l in labels[i]:
                entities[l].add(resource)

    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

