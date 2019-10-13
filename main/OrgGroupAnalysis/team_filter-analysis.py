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
    MAX_ITER = 10
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
    iteration = 1
    discarded_resources_singleton = list()
    discarded_resources_negative = list()
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
            pct_r_left_global = ((len(set(log['resource'])) - len(l_r_rm))
                / num_resources_total)
            pct_case_rm = len(l_case_rm) / len(set(log['case_id']))
            pct_case_left_global = ((len(set(log['case_id'])) - len(l_case_rm))
                / num_cases_total)

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
                pct_case_rm,
                pct_r_left_global, 
                pct_case_left_global 
                ))

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
            print('Select a "K":')
            print('\tfor proceeding to the next iteration, type in the'
                'number;')
            print('\tfor generating a final decision and stopping iteration,'
                'type in the number ended with a hash "#", e.g. 5#')
            print('K = ', end='')
            next_k = input()
            if next_k.endswith('#'):
                next_k = int(next_k[:-1])
                stop_iteration = True
            else:
                next_k = int(next_k)
                stop_iteration = False

            ogs, _ = _ahc(profiles, next_k)
            if stop_iteration:
                for i, og in enumerate(ogs):
                    print('Group {}:'.format(i))
                    print(profiles.loc[sorted(list(og))])
                iteration = MAX_ITER
            else:
                scores = silhouette_score(ogs, profiles)
                l_r_rm = set(r for r in scores if scores[r] <= 0.0)
                discarded_resources_singleton.append(
                    set(r for r in scores if scores[r] == 0.0))
                discarded_resources_negative.append(
                    set(r for r in scores if scores[r] < 0.0))
                l_case_rm = list()
                for case_id, events in log.groupby('case_id'):
                    resources = set(events['resource'])
                    if len(l_r_rm.intersection(resources)) > 0:
                        l_case_rm.append(case_id)
                # filter out cases and generate a new log
                log = log[log['case_id'].map(lambda x: x not in l_case_rm)]

        else:
            print('\nMAX iteration reached:')

        # proceed to the next iteration
        iteration += 1
    
    print('-' * 80)
    total_rl = mode_miner.derive_resource_log(el)
    total_profiles = count_execution_frequency(total_rl, scale='log')
    print('Singletons discarded:')
    for it, l_r_rm in enumerate(discarded_resources_singleton):
        print('at Iteration {}:'.format(it+1))
        print(total_profiles.loc[sorted(list(l_r_rm))])
    print()
    print('Negatives discarded:')
    for it, l_r_rm in enumerate(discarded_resources_negative):
        print('at Iteration {}:'.format(it+1))
        print(total_profiles.loc[sorted(list(l_r_rm))])

    # TODO: treat the negatives as of activity-based?
    profiles_act = count_execution_frequency(
        ATonlyMiner(el).derive_resource_log(el),
        scale='log').loc[list(set.union(*discarded_resources_negative))]
    if len(profiles_act) >= 3:
        for k in range(2, len(profiles_act)):
            ogs_negatives, _ = _ahc(
                profiles_act, k)
            scores_negatives = silhouette_score(
                ogs_negatives, profiles_act)
            print('Silhouette scores (assuming ACT-based) for k={} is\t{}'.format(
                k, mean(list(scores_negatives.values()))))
    else:
        print(profiles_act)

