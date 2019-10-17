#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program is written to perform an auto DFS based on the original program
stored in "team_filter-analysis.py" (so as to make life easier). All possible
successful "paths" found will be delivered as output.
'''

import sys
sys.path.append('./src/')

from ExecutionModeMiner.direct_groupby import ATonlyMiner, CTonlyMiner
from ExecutionModeMiner.informed_groupby import TraceClusteringCTMiner
from ResourceProfiler.raw_profiler import count_execution_frequency
from OrganizationalModelMiner.clustering.hierarchical import _ahc
from Evaluation.m2m.cluster_validation import silhouette_score
from numpy import mean, amin, amax

fn_event_log = sys.argv[1]

def foo(mode_miner, num_groups, log, path_str):
    rl = mode_miner.derive_resource_log(log)
    profiles = count_execution_frequency(rl, scale='log')

    valid_selection = list()
    for k in num_groups:
        # calculate silhouette scores and variance explained (for each K)
        ogs, _ = _ahc(profiles, k)
        scores = silhouette_score(ogs, profiles)

        # analyze the required items (for each K)
        # object-level
        mean_score_overall = mean(
            list(scores.values()))
        num_singleton_clusters = sum(
            [1 for x in scores.values() if x == 0.0])
        k_prime = k - num_singleton_clusters

        # cluster-level
        scores_clu = list()
        for g in ogs:
            if len(g) > 1:
                score_g = mean([scores[r]
                    for r in g if scores[r] != 0.0])
                max_score_g = amax([scores[r]
                    for r in g if scores[r] != 0.0])
                scores_clu.append((score_g, max_score_g))
        k_flag = all(
            [(x[1] >= mean_score_overall) for x in scores_clu])

        # amount of resources, cases to be discarded (for each K)
        l_r_rm = set(r for r in scores if scores[r] <= 0.0)
        l_case_rm = list()
        for case_id, events in log.groupby('case_id'):
            resources = set(events['resource'])
            if len(l_r_rm.intersection(resources)) > 0:
                l_case_rm.append(case_id)
        pct_case_rm = len(l_case_rm) / len(set(log['case_id']))

        if k_flag and k_prime == 3 and pct_case_rm < 0.25:
            if len(l_case_rm) == 0:
                # a valid leaf found
                print()
                print('-' * 10 + ' Found valid solution: {} '.format(
                    path_str + str(k) + '.') + '-' * 10)
                print()
            else:
                # proceed to next recursion
                foo(mode_miner, num_groups,
                    log[log['case_id'].map(lambda x: x not in l_case_rm)],
                    path_str + str(k) + ',')
        else:
            # do nothing (not care about invalid leaves)
            pass

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)

    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={'(case) channel': 6}) # wabo
        #el = read_disco_csv(f, mapping={'(case) channel': 6}) # bpic12 TODO
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 7}) # bpic17

    num_groups = list(range(2, 10))

    #mode_miner = ATonlyMiner(el)
    mode_miner = CTonlyMiner(el, case_attr_name='(case) channel')
    #mode_miner = TraceClusteringCTMiner(el, fn_partition=fn_partition)

    '''
        Each of the particular selection of "k" is evaluated by both:
        - a value resulted from measuring by elbow method, cf.
            (https://en.wikipedia.org/wiki/Elbow_method_(clustering))
        - decision drawn from silhouette analysis, cf.
            (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
    '''
    ''' {12} items are collected for analysis:
        - the result from silhouette analysis (k_flag),
        - the average value of silhouette score (of all resources), excluding
          those designated to singleton clusters;
        - the value from elbow method (cf. TODO) (of the whole clustering);
        - the number of singleton clusters;
        - the number of resources with POSITIVE silhouette score;
        - the number of resources with NEGATIVE silhouette score;
        - the number of clusters with POS average silhouette score (excluding
          singletons);
        - the number of clusters with NEG average silhouette score (excluding
          singletons);
        - percentage of resources having zero/negative silhouette scores;
        - percentage of cases involving the resouces above;
        - percentage of resources having zero/negative silhouette scores (global);
        - percentage of cases involving the resouces above (global);
    '''

    foo(mode_miner, num_groups, el, '')

