#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import copy
from collections import defaultdict

# Joint cases
def SAR(cases):
    print('Working together: simultaneous appearance')
    cnt = 0
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for caseid, trace in cases.items():
        cnt += 1
        res_all = list(set([tr[2] for tr in trace]))
        for i in range(len(res_all) - 1):
            for j in range(i+1, len(res_all)):
                mat[res_all[i]][res_all[j]] += 1
                mat[res_all[j]][res_all[i]] += 1

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

