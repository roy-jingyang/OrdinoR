#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import copy
from collections import defaultdict

def CCCDCM(cases):
    print('Handover: consider causality, consider direct succession, consider multiple apperance.')
    cnt = 0
    # TODO: non task-specific now
    # TODO: non directed now
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace) - 1):
            res_prev = trace[i][2]
            res_next = trace[i+1][2]
            dur = trace[i+1][-2] - trace[i][-1]
            mat[res_prev][res_next] += dur.total_seconds()
            mat[res_next][res_prev] += dur.total_seconds()

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

def CCCDIM(cases):
    pass

