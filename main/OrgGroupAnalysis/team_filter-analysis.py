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

    MAX_ITER = 2

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
    ''' {10} items are collected for analysis:
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
        'pct_case_to_be_filtered'
    ]
    from Evaluation.m2m.cluster_validation import silhouette_score
    from Evaluation.m2m.cluster_validation import variance_explained_percentage
    from numpy import mean, amin, amax

    log = el
    iteration = 1
    while iteration <= MAX_ITER:
        print('-' * 30 + 'Iteration: {}'.format(iteration) + '-' * 30)

        rl = mode_miner.derive_resource_log(log)
        profiles = count_execution_frequency(rl, scale='log')
        l_measured_values = list()
        for k in num_groups:
            # calculate silhouette scores and variance explained (for each K)
            ogs, _ = _ahc(profiles, k)
            scores = silhouette_score(ogs, profiles)
            var_pct = variance_explained_percentage(
                ogs, profiles)

            # analyze the required items (for each K)
            # object-level
            mean_score_overall = mean(
                list(scores.values()))
            num_singleton_clusters = sum(
                [1 for x in scores.values() if x == 0.0])
            num_pos_score_objects = sum(
                [1 for x in scores.values() if x > 0.0])
            num_neg_score_objects = sum(
                [1 for x in scores.values() if x < 0.0])

            # cluster-level
            scores_clu = list()
            for g in ogs:
                if len(g) > 1:
                    score_g = mean([scores[r]
                        for r in g if scores[r] != 0.0])
                    max_score_g = amax([scores[r]
                        for r in g if scores[r] != 0.0])
                    scores_clu.append((score_g, max_score_g))
            num_pos_score_clusters = sum(
                [1 for x in scores_clu if x[0] > 0])
            num_neg_score_clusters = sum(
                [1 for x in scores_clu if x[0] <= 0])
            k_flag = all(
                [(x[1] >= mean_score_overall) for x in scores_clu])

            # amount of resources, cases to be discarded (for each K)
            l_r_rm = set(r for r in scores if scores[r] <= 0.0)
            l_case_rm = list()
            for case_id, events in log.groupby('case_id'):
                resources = set(events['resource'])
                if len(l_r_rm.intersection(resources)) > 0:
                    l_case_rm.append(case_id)
            pct_r_rm = len(l_r_rm) / len(set(log['resource']))
            pct_case_rm = len(l_case_rm) / len(set(log['case_id']))

            l_measured_values.append((
                k_flag,
                mean_score_overall,
                var_pct,
                num_singleton_clusters,
                num_pos_score_objects,
                num_neg_score_objects,
                num_pos_score_clusters,
                num_neg_score_clusters,
                pct_r_rm,
                pct_case_rm))

        print('VALUES of K')
        print(','.join(str(k) for k in num_groups))
        print()
        for i, attribute in enumerate(items): 
            print(attribute)
            print(','.join(str(l_measured_values[ik][i])
                for ik, k in enumerate(num_groups)))
            print()

        # prompt for selecting a 'K'
        if iteration + 1 <= MAX_ITER:
            print('Select a "K" for the next iteration:\t', end='')
            next_k = int(input())
            ogs, _ = _ahc(profiles, next_k)
            scores = silhouette_score(ogs, profiles)
            l_r_rm = set(r for r in scores if scores[r] <= 0.0)
            l_case_rm = list()
            for case_id, events in log.groupby('case_id'):
                resources = set(events['resource'])
                if len(l_r_rm.intersection(resources)) > 0:
                    l_case_rm.append(case_id)
            # filter out cases and generate a new log
            log = log[log['case_id'].map(lambda x: x not in l_case_rm)]

        # proceed to the next iteration
        iteration += 1

