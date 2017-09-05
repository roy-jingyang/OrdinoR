#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import copy
from collections import defaultdict

def SAR(cases):
    print('Working together: simultaneous appearance')
    cnt = 0
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for caseid, trace in cases.items():
        cnt += 1
        case_dur = trace[-1][-1] - trace[0][-2]
        res_all = [tr[2] for tr in trace]
        for i in range(len(res_all) - 1):
            for j in range(i+1, len(res_all)):
                mat[res_all[i]][res_all[j]] = case_dur.total_seconds()
                mat[res_all[j]][res_all[i]] = case_dur.total_seconds()

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

