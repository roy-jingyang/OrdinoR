#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import copy
from collections import defaultdict

# Warning: Including real causality into consideration requires support of the process model
def CCCDCM(cases, depth=1, beta=1):
    print('# TODO')
    print('Handover of work: consider real causality, consider direct succession, consider multiple appearance.')
    return None

# Warning: Including real causality into consideration requires support of the process model
def CCCDIM(cases, depth=1, beta=1):
    print('# TODO')
    print('Handover of work: consider real causality, consider direct succession, ignore multiple appearance.')
    return None

def ICCDCM(cases, depth=1, beta=1):
    print('Handover of work: ignore real causality, consider direct succession, consider multiple appearance.')
    cnt = 0
    # TODO: non task-specific now
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: SIGMA_Case c in Log (|c| - 1)
    scale_factor = 0
    for caseid, trace in cases.items():
        scale_factor += len(trace) - 1

    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace) - 1):
            # within a case
            res_prev = trace[i][2]
            res_next = trace[i+1][2]
            #dur = trace[i+1][-2] - trace[i][-1]
            #mat[res_prev][res_next] += dur.total_seconds() / scale_factor
            mat[res_prev][res_next] += 1 / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

# Warning: Ignoring multiple appearance does NOT apply to time-based calculation
def ICCDIM(cases, depth=1, beta=1):
    print('Handover of work: ignore real causality, consider direct succession, ignore multiple appearance.')
    cnt = 0
    # TODO: non task-specific now
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: |L|
    scale_factor = len(cases)

    for caseid, trace in cases.items():
        cnt += 1
        handovered_dyads = set()
        for i in range(len(trace) - 1):
            # within a case
            res_prev = trace[i][2]
            res_next = trace[i+1][2]
            handovered_dyads.add((res_prev, res_next))
        for dyad in handovered_dyads:
            mat[dyad[0]][dyad[1]] += 1 / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

def ICCICM(cases, depth, beta):
    print('Handover of work: ignore real causality, consider indirect succession, consider multiple appearance.')
    cnt = 0
    # TODO: non task-specific now
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: SIGMA_Case c in Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1 * (|c| - n)))
    scale_factor = 0
    for caseid, trace in cases.items():
        for n in range(1, min(len(trace) - 1, depth)):
            scale_factor += beta ** (n - 1) * (len(trace) - n)

    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace) - 1):
            # within a case
            for n in range(1, min(len(trace) - 1, depth)):
                # for each calculation depth level
                if i + n > len(trace) - 1:
                    pass
                else:
                    res_prev = trace[i][2]
                    res_next = trace[i+n][2]
                    #dur = trace[i+1][-2] - trace[i][-1]
                    #mat[res_prev][res_next] += beta ** (n - 1) * dur.total_seconds() / scale_factor
                    mat[res_prev][res_next] += beta ** (n - 1) * 1 / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

# Warning: Ignoring multiple appearance does NOT apply to time-based calculation
def ICCIIM(cases, depth, beta):
    print('Handover of work: ignore real causality, consider indirect succession, ignore multiple appearance.')
    cnt = 0
    # TODO: non task-specific now
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: SIGMA_Case c in Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1))
    scale_factor = 0
    for caseid, trace in cases.items():
        for n in range(1, min(len(trace) - 1, depth)):
            scale_factor += beta ** (n - 1)

    for caseid, trace in cases.items():
        cnt += 1
        handovered_dyads = set()
        for i in range(len(trace) - 1):
            # within a case
            for n in range(1, min(len(trace) - 1, depth)):
                # for each calculation depth level
                if i + n > len(trace) - 1:
                    pass
                else:
                    res_prev = trace[i][2]
                    res_next = trace[i+n][2]
                    handovered_dyads.add((res_prev, res_next, beta ** (n - 1)))
        for dyad in handovered_dyads:
            mat[dyad[0]][dyad[1]] += dyad[2] * 1 / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

