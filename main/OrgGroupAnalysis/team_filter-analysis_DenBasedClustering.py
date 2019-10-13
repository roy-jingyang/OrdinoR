#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)

    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={'(case) channel': 6}) # wabo
        #el = read_disco_csv(f, mapping={'(case) channel': 6}) # bpic12 TODO
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 7}) # bpic17

    num_groups = list(range(3, 17))
    #FLAG_ANALYSIS_OBJECTS = "kept"
    #FLAG_ANALYSIS_OBJECTS = "discarded"

    from ExecutionModeMiner.direct_groupby import ATonlyMiner, CTonlyMiner
    from ExecutionModeMiner.informed_groupby import TraceClusteringCTMiner
    from ResourceProfiler.raw_profiler import count_execution_frequency
    from OrganizationalModelMiner.clustering.hierarchical import _ahc

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
    items = [
        'VALID_K_value',
        'average_silhouette_score_overall',
        'value_elbow_method',
        'num_singletons',
        'num_res_pos_silhouette_score',
        'num_res_neg_silhouette_score',
        'num_clu_pos_silhouette_score',
        'num_clu_neg_silhouette_score',
        'pct_resource_to_be_filtered',
        'pct_case_to_be_filtered',
        'pct_resource_to_be_left__global',
        'pct_case_to_be_left_global'
    ]
    from Evaluation.m2m.cluster_validation import silhouette_score
    from Evaluation.m2m.cluster_validation import variance_explained_percentage
    from numpy import mean, amin, amax

    num_resources_total = len(set(el['resource']))
    num_cases_total = len(set(el['case_id']))

    log = el
    proceed = True
    while proceed:
        rl = mode_miner.derive_resource_log(log)
        profiles = count_execution_frequency(rl, scale='log')

        from sklearn.cluster import DBSCAN, OPTICS
        from collections import defaultdict
        X = profiles.values
        #labels = DBSCAN(min_samples=3, metric='euclidean').fit_predict(X)
        labels = OPTICS(min_samples=3, p=4).fit_predict(X)

        groups = defaultdict(set)
        outliers = set()
        for ir, r in enumerate(profiles.index):
            if labels[ir] == -1:
                outliers.add(r)
            else:
                groups[labels[ir]].add(r)
        ogs = [frozenset(g) for g in groups.values()]

        for i, og in enumerate(ogs):
            print('Group {}:'.format(i))
            print(profiles.loc[sorted(list(og))])

        print('-' * 30 + 'Outliers' + '-' * 30)
        print(profiles.loc[sorted(list(outliers))])

        print('Proceed (Y/n)?', end='')
        option = input()
        proceed = True if option == 'Y' or option == '' else False

        l_case_rm = list()
        for case_id, events in log.groupby('case_id'):
            resources = set(events['resource'])
            if len(outliers.intersection(resources)) > 0:
                l_case_rm.append(case_id)

        log = log[log['case_id'].map(lambda x: x not in l_case_rm)]

