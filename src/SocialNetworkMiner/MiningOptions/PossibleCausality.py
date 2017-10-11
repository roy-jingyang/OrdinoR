#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import defaultdict

# Warning: Including real causality into consideration requires support of the process model

# Frequency-based
def handover_CDCM(cases, is_task_specific=False, depth=1):
    print('Possible Causality: Handover of work (Ignore real Causality,
            Consider Direct succession, Consider Multiple appearance.)')
    cnt = 0
    if is_task_specific:
        mat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
    else:
        mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: SIGMA_Case c in Log (|c| - 1)
    scale_factor = 0
    for caseid, trace in cases.items():
        scale_factor += len(trace) - 1

    for caseid, trace in cases.items():
        cnt += 1
        frequency = 0
        for i in range(len(trace) - 1):
            # within a case
            res_prev = trace[i][2]
            res_next = trace[i+1][2]
            if is_task_specific:
                act_prev = trace[i][1]
                act_next = trace[i+1][1]
                mat[act_prev][act_next][res_prev][res_next] += 1 / scale_factor
            else:
                mat[res_prev][res_next] += 1 / scale_factor

# Duration-time-based
def handover_duration(cases, is_task_specific=False, depth=1):
    print('Possible Causality: Handover duration of work (Ignore real Causality,
            Consider Direct succession, Consider Multiple appearance.)')
    cnt = 0
    if is_task_specific:
        mat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))))
    else:
        mat = defaultdict(lambda: defaultdict(lambda: 0))
    # scale_factor: SIGMA_Case c in Log (|c| - 1)
    scale_factor = 0
    for caseid, trace in cases.items():
        scale_factor += len(trace) - 1

    for caseid, trace in cases.items():
        cnt += 1
        frequency = 0
        for i in range(len(trace) - 1):
            # within a case
            res_prev = trace[i][2]
            res_next = trace[i+1][2]
            dur = trace[i+1][-2] - trace[i][-1]
            if is_task_specific:
                act_prev = trace[i][1]
                act_next = trace[i+1][1]
                mat[act_prev][act_next][res_prev][res_next] += dur.total_seconds() / scale_factor
            else:
                mat[res_prev][res_next] += dur.total_seconds() / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)

# Frequency-based
def handover_CDIM(cases, depth=1):
    print('Possible Causality: Handover of work (Ignore real Causality,
            Consider Direct succession, Ignore Multiple appearance.)')
    cnt = 0
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

# Frequency-based
def handover_CICM(cases, depth, beta):
    print('Possible Causality: Handover of work (Ignore real Causality,
            Consider Indirect succession, Consider Multiple appearance.)')
    cnt = 0
    cnt = 0
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
                    mat[res_prev][res_next] += beta ** (n - 1) * 1 / scale_factor

    print('# of cases processed: {}'.format(cnt))
    return copy.deepcopy(mat)


# Frequency-based
def handover_CIIM(cases, depth, beta):
    print('Possible Causality: Handover of work (Ignore real Causality,
            Consider Indirect succession, Ignore Multiple appearance.)')
    cnt = 0
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

