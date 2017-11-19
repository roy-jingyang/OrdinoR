#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import defaultdict
import numpy as np

# Joint activities

def _JointActivities_Base(cases):
    cnt = 0
    #counting_mat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = []

    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace)):
            res = trace[i][2]
            activity = trace[i][1]
            if activity in activity_index:
                pass
            else:
                activity_index.append(activity)
            #proc_dur = trace[i][5] - trace[i][4]
            #counting_mat[res][activity][0] += 1
            #counting_mat[res][activity][1] += proc_dur.total_seconds()
            counting_mat[res][activity] += 1
    print('# of cases processed: {}'.format(cnt))
    profile_mat = defaultdict(lambda: [0] * len(activity_index))
    for res, activities in counting_mat.items():
        for act, count in activities.items():
            index = activity_index.index(act)
            #profile_mat[res][index] = counts[1] / counts[0]
            profile_mat[res][index] = count

    return copy.deepcopy(profile_mat)

def EuclideanDist(cases):
    print('Metric based on Joint Activities: Euclidean Distance')
    profile_mat = _JointActivities_Base(cases)
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(profile_mat.keys()) - 1):
        for j in range(i+1, len(profile_mat.keys())):
            res_i = list(profile_mat.keys())[i]
            res_j = list(profile_mat.keys())[j]
            profile_i = profile_mat[res_i]
            profile_j = profile_mat[res_j]

            euclidean_dist = np.linalg.norm(
                    np.array(profile_i, dtype=np.float64) -
                    np.array(profile_j, dtype=np.float64))
            mat[res_i][res_j] = euclidean_dist
            mat[res_j][res_i] = euclidean_dist

    return copy.deepcopy(mat)

def CorrelationCoefficient(cases, threshold_top_pct):
    print('Metric based on Joint Activities: Correlation (PCC)')
    from scipy.stats import pearsonr
    profile_mat = _JointActivities_Base(cases)
    resource_index = list(profile_mat.keys())
    mat_pcc = np.zeros((len(profile_mat.keys()), len(profile_mat.keys())))
    for i in range(len(resource_index) - 1):
        for j in range(i + 1, len(resource_index)):
            res_i = resource_index[i]
            res_j = resource_index[j]
            profile_i = profile_mat[res_i]
            profile_j = profile_mat[res_j]

            cc = pearsonr(
                    np.array(profile_i, dtype=np.float64),
                    np.array(profile_j, dtype=np.float64))
            mat_pcc[i][j] = cc[0]
            mat_pcc[j][i] = cc[0]

    threshold = np.percentile(mat_pcc[mat_pcc > 0.0], 
            (1 - threshold_top_pct) * 100)
    #mat = defaultdict(lambda: defaultdict(lambda: 0))
    mat = defaultdict(lambda: defaultdict(lambda: None))
    for i in range(len(resource_index) - 1):
        for j in range(i + 1, len(resource_index)):
            res_i = resource_index[i]
            res_j = resource_index[j]

            if mat_pcc[i][j] >= threshold:
                mat[res_i][res_j] = mat_pcc[i][j]
                mat[res_j][res_i] = mat_pcc[j][i]
            else:
                mat[res_i][res_j] = None
                mat[res_j][res_i] = None

    return copy.deepcopy(mat)

