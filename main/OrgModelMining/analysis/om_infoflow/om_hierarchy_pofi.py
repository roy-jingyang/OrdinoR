#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fn_org_model_hcy = sys.argv[2]
fnout_result = sys.argv[3]

if __name__ == '__main__':
    from pandas import read_csv
    og_hcy = read_csv(fn_org_model_hcy, header=0, index_col=0)
    n_resources = len(og_hcy.index)
    n_levels = len(og_hcy.columns)

    # calculate the distance between any pair of resources given the hierarchy
    from collections import defaultdict
    hcy_distance = defaultdict(lambda: defaultdict(lambda: None))
    from itertools import combinations
    for pair in combinations(og_hcy.index, 2):
        # scan from right to left to find the common ancestor node
        u = pair[0]
        v = pair[1]
        distance = 0
        for j in reversed(range(n_levels)):
            if og_hcy.loc[u][j] != og_hcy.loc[v][j]:
                distance += 1
            else:
                hcy_distance[u][v] = hcy_distance[v][u] = \
                        distance / n_levels
                break

    # calculate the (normalized) index using the event log
    pofi_values = list()
    from orgminer.IO.reader import read_disco_csv
    cases = read_disco_csv(fn_event_log)
    for case_id, trace in cases.groupby('case_id'):
        pofi_case = 0
        for i in range(len(trace) - 1):
            res_prev = trace.iloc[i]['resource']
            res_next = trace.iloc[i + 1]['resource']
            if res_prev != res_next: # self-loop ignored
                pofi_case += hcy_distance[res_prev][res_next]
        if len(trace) > 1:
            pofi_values.append((case_id, pofi_case / (len(trace) - 1)))
        else:
            pofi_values.append((case_id, 0))

    from csv import writer
    with open(fnout_result, 'w') as f:
        writer(f).writerow(['case_id', 'pofi (case)'])
        writer(f).writerows(pofi_values)

