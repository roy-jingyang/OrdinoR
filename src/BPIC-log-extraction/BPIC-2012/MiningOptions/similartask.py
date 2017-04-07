#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import copy
from collections import defaultdict
from numpy import array, linalg

def _SimilarTask_Base(cases):
    cnt = 0
    counting_mat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    activity_index = ['W_Valideren aanvraag', 'W_Nabellen offertes', 'W_Nabellen incomplete dossiers', 'W_Afhandelen leads', 'W_Completeren aanvraag']
    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace)):
            res = trace[i][2]
            activity = trace[i][3]
            proc_dur = trace[i][5] - trace[i][4]
            counting_mat[res][activity][0] += 1
            counting_mat[res][activity][1] += proc_dur.total_seconds()
    print('# of cases processed: {}'.format(cnt))
    profile_mat = defaultdict(lambda: [0] * 5)
    for res, activities in counting_mat.items():
        for act, counts in activities.items():
            index = activity_index.index(act)
            profile_mat[res][index] = counts[1] / counts[0]

    return copy.deepcopy(profile_mat)

def ED(cases):
    print('Similar task: ')
    profile_mat = _SimilarTask_Base(cases)
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for i in range(len(profile_mat.keys()) - 1):
        for j in range(i+1, len(profile_mat.keys())):
            res_i = list(profile_mat.keys())[i]
            res_j = list(profile_mat.keys())[j]
            euclidean_dist = linalg.norm(array(res_i) - array(res_j))
            mat[res_i][res_j] = euclidean_dist
            mat[res_j][res_i] = euclidean_dist

    return copy.deepcopy(mat)

