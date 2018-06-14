#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import defaultdict
import numpy as np

# Joint activities
def mine(cases, dist_type, threshold_top_pct=1.0):
    print('Metrics based on Joint Activities: {}'.format(dist_type))
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

    # adjacency-matrix for performer-activity
    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count


    # TODO: logarithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    # build resource social network G (with linear tf.) with same indexing
    # TODO: Euclidean Distance / PCC Dist ('Correlation distance')
    from scipy.spatial.distance import squareform, pdist
    if dist_type == 'euclidean':
        # Distance -> Similarity
        G = squareform(pdist(profile_mat, metric='euclidean'))
        G = 1 - ((G - G.min()) / (G.max() - G.min())) # 1 - MinMaxScaled
    elif dist_type == 'pcc':
        # Distance -> Correlation
        G = squareform(pdist(profile_mat, metric='correlation'))
        G = 1 - G
    else:
        return None

    # take only the positive values (for pcc)
    threshold = np.percentile(G[G > 0.0], (1 - threshold_top_pct) * 100)

    mat = defaultdict(lambda: defaultdict(lambda: None))
    for i in range(len(resource_index) - 1):
        for j in range(i + 1, len(resource_index)):
            res_i = resource_index[i]
            res_j = resource_index[j]

            if G[i][j] >= threshold:
                mat[res_i][res_j] = G[i][j]
                mat[res_j][res_i] = G[j][i]
            else:
                mat[res_i][res_j] = None
                mat[res_j][res_i] = None

    return copy.deepcopy(mat)

