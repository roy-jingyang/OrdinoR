#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
#fn_report = sys.argv[2]

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)

    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={'(case) channel': 6}) # wabo
        #el = read_disco_csv(f, mapping={'(case) channel': 6}) # bpic12 TODO
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 7}) # bpic17

    print('Set K = ', end='')
    num_groups = int(input())
    #num_groups = list(range(3, 14))

    from ExecutionModeMiner.direct_groupby import ATonlyMiner, CTonlyMiner
    from ExecutionModeMiner.informed_groupby import TraceClusteringCTMiner
    from ResourceProfiler.raw_profiler import count_execution_frequency
    from OrganizationalModelMiner.clustering.hierarchical import _ahc

    mode_miner = ATonlyMiner(el)
    #mode_miner = CTonlyMiner(el, case_attr_name='(case) channel')
    #mode_miner = TraceClusteringCTMiner(el, fn_partition=fn_partition)

    rl = mode_miner.derive_resource_log(el)
    profiles = count_execution_frequency(rl, scale='log')

    '''
        Each of the particular selection of "k" is evaluated by both:
        - a value resulted from measuring by elbow method, cf.
            (https://en.wikipedia.org/wiki/Elbow_method_(clustering))
        - decision drawn from silhouette analysis, cf.
            (https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
    '''
    ''' {7} items are collected for analysis:
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
    '''
    items = [
        'average_silhouette_score_overall',
        'value_elbow_method',
        'num_singletons',
        'num_res_pos_silhouette_score',
        'num_res_neg_silhouette_score',
        'num_clu_pos_silhouette_score',
        'num_clu_neg_silhouette_score',
    ]
    from Evaluation.m2m.cluster_validation import silhouette_score
    from Evaluation.m2m.cluster_validation import variance_explained_percentage
    from numpy import mean, amin, amax

    '''
    max_gain = 0
    max_after_filtered = 0
    # Test: {
    for k in num_groups:
        ogs, _ = _ahc(profiles, k)
        scores = silhouette_score(ogs, profiles)
        mean_score_overall_old = mean(
            list(scores.values()))

        #l_r_rm = set(r for r in scores if scores[r] == 0.0)
        l_r_rm = set(r for r in scores if scores[r] <= 0.0)
        print('\tFiltering out {} ({:.0%} of all {}) resources.'.format(
            len(l_r_rm),
            (len(l_r_rm) / len(set(el['resource']))),
            len(set(el['resource']))))
        del scores

        # Discovery (2): re-run on filtered log (discard cases)
        l_case_rm = list()
        for case_id, events in el.groupby('case_id'):
            resources = set(events['resource'])
            if len(l_r_rm.intersection(resources)) > 0:
                l_case_rm.append(case_id)
        print('\tFiltering out {} ({:.0%} of all {}) cases.'.format(
            len(l_case_rm),
            (len(l_case_rm) / len(set(el['case_id']))),
            len(set(el['case_id']))))

        el_new = el[el['case_id'].map(lambda x: x not in l_case_rm)]
        rl_new = mode_miner.derive_resource_log(el_new)
        profiles_new = count_execution_frequency(rl_new, scale='log')
        ogs_filtered, _ = _ahc(profiles_new, k)
        scores = silhouette_score(ogs_filtered, profiles_new)

        # Analysis
        mean_score_overall = mean(
            #[x for x in scores.values() if x != 'nan'])
            list(scores.values()))
        max_after_filtered = (mean_score_overall 
            if mean_score_overall > max_after_filtered else max_after_filtered)

        gain = mean_score_overall - mean_score_overall_old
        print('K = {}:\t{}\t{}\t(delta = {})'.format(
            k, True if gain > 0 else False, mean_score_overall, gain))
        max_gain = gain if gain > max_gain else max_gain
    print(max_gain)
    print(max_after_filtered)
    exit()
    # }
    '''

    l_measured_values = list()
    # Discovery (1): identify singletons and filter out
    ogs, _ = _ahc(profiles, num_groups)
    scores = silhouette_score(ogs, profiles)
    print('{} to be filtered out:'.format(
        sum([1 for x in scores.values() if x <= 0.0])))
    for r, score in scores.items():
        if score <= 0.0:
            print('\t', end='')
            print(r)

    #l_r_rm = set(r for r in scores if scores[r] == 0.0)
    l_r_rm = set(r for r in scores if scores[r] <= 0.0)
    del scores

    # Discovery (2): re-run on filtered log (discard cases)
    l_case_rm = list()
    for case_id, events in el.groupby('case_id'):
        resources = set(events['resource'])
        if len(l_r_rm.intersection(resources)) > 0:
            l_case_rm.append(case_id)

    print('\nFiltering out {} ({:.0%} of all {}) cases.'.format(
        len(l_case_rm),
        (len(l_case_rm) / len(set(el['case_id']))),
        len(set(el['case_id']))))

    el_good = el[el['case_id'].map(lambda x: x not in l_case_rm)]
    rl_good = mode_miner.derive_resource_log(el_good)
    profiles_good = count_execution_frequency(rl_good, scale='log')
    ogs_filtered, _ = _ahc(profiles_good, num_groups)
    scores = silhouette_score(ogs_filtered, profiles_good)
    var_explained_pct = variance_explained_percentage(
        ogs_filtered, profiles_good)

    # Analysis
    mean_score_overall = mean(
        #[x for x in scores.values() if x != 'nan'])
        list(scores.values()))
    num_singleton_clusters = sum(
        #[1 for x in scores.values() if x == 'nan'])
        [1 for x in scores.values() if x == 0.0])
    num_pos_score_objects = sum(
        #[1 for x in scores.values() if x != 'nan' and x > 0.0])
        [1 for x in scores.values() if x > 0.0])
    num_neg_score_objects = sum(
        #[1 for x in scores.values() if x != 'nan' and x <= 0.0])
        [1 for x in scores.values() if x < 0.0])

    scores_clu = list()
    for g in ogs_filtered:
        if len(g) > 1:
            score_g = mean([scores[r] 
                #for r in g if scores[r] != 'nan'])
                for r in g if scores[r] != 0.0])
            max_score_g = amax([scores[r]
                #for r in g if scores[r] != 'nan'])
                for r in g if scores[r] != 0.0])
            scores_clu.append((score_g, max_score_g))
    num_pos_score_clusters = sum(
        [1 for x in scores_clu if x[0] > 0])
    num_neg_score_clusters = sum(
        [1 for x in scores_clu if x[0] <= 0])
    k_flag = all(
        [(x[1] >= mean_score_overall) for x in scores_clu])

    print((
        mean_score_overall,
        var_explained_pct,
        num_singleton_clusters,
        num_pos_score_objects,
        num_neg_score_objects,
        num_pos_score_clusters,
        num_neg_score_clusters,
        k_flag
    ))

    del scores

    el_worse = el[el['case_id'].map(lambda x: x in l_case_rm)]
    #mode_miner = ATonlyMiner(el)
    mode_miner = CTonlyMiner(el, case_attr_name='(case) channel')
    rl_worse = mode_miner.derive_resource_log(el_worse)
    profiles_worse = count_execution_frequency(rl_worse, scale='log')

    num_groups = list(range(3, 14))
    for k in num_groups:
        ogs, _ = _ahc(profiles_worse, k)
        scores = silhouette_score(ogs, profiles_worse)

        # Analysis
        mean_score_overall = mean(
            #[x for x in scores.values() if x != 'nan'])
            list(scores.values()))
        print('K = {}:\t{}'.format(k, mean_score_overall))

