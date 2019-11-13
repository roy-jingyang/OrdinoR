#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

fn_event_log = sys.argv[1]
fn_report = sys.argv[2]

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)

    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={'(case) channel': 6}) # wabo
        #el = read_disco_csv(f, mapping={'(case) channel': 6}) # bpic12 TODO
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 7}) # bpic17

    num_groups = list(range(3, 17)) # wabo: k \in [3, 17)
    #num_groups = list(range(3, 24)) # bpic12: k \in [3, 14)
    #num_groups = list(range(3, 49)) # bpic17: k \in [3, 49)

    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner, CTonlyMiner
    from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringCTMiner
    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    from orgminer.OrganizationalModelMiner.clustering.hierarchical import _ahc

    # Mining and Evaluating activity-based teams
    act_based_miner = ATonlyMiner(el)
    rl_act = act_based_miner.derive_resource_log(el)
    profiles_act = count_execution_frequency(rl_act, scale='log')

    # Mining and Evaluating case-based teams: case attributes
    case_based_miner = CTonlyMiner(el, case_attr_name='(case) channel')
    #case_based_miner = CTonlyMiner(el, case_attr_name='(case) channel')
    #case_based_miner = CTonlyMiner(el, case_attr_name='(case) LoanGoal')
    rl_case = case_based_miner.derive_resource_log(el)
    profiles_case = count_execution_frequency(rl_case, scale='log')

    '''
    # Mining and Evaluating case-based teams (1): trace clustering
    print('Input the path of the partitioning file:', end=' ')
    fn_partition = input()
    case_based_miner1 = TraceClusteringCTMiner(el, fn_partition=fn_partition)
    rl_case1 = case_based_miner1.derive_resource_log(el)
    profiles_case1 = count_execution_frequency(rl_case1, scale='log')
    '''

    # Discovery and Analysis
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

    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from orgminer.Evaluation.m2m.cluster_validation import variance_explained_percentage
    from numpy import mean, amin, amax
    l_measured_values_act = list()
    l_measured_values_case = list()
    #l_measured_values_case1 = list()

    for k in num_groups:
        ogs_act, _ = _ahc(profiles_act, k)
        scores_act = silhouette_score(ogs_act, profiles_act)
        variance_explained_percentage_act = variance_explained_percentage(
            ogs_act, profiles_act)

        ogs_case, _ = _ahc(profiles_case, k)
        scores_case = silhouette_score(ogs_case, profiles_case)
        variance_explained_percentage_case = variance_explained_percentage(
            ogs_case, profiles_case)

        '''
        ogs_case1, _ = _ahc(profiles_case1, k)
        scores_case1 = silhouette_score(ogs_case1, profiles_case1)
        variance_explained_percentage_case1 = variance_explained_percentage(
            ogs_case1, profiles_case1)
        '''

        # ACT object level
        mean_score_overall_act = mean(
            #[x for x in scores_act.values() if x != 'nan'])
            list(scores_act.values()))
        num_singleton_clusters_act = sum(
            #[1 for x in scores_act.values() if x == 'nan'])
            [1 for x in scores_act.values() if x == 0.0])
        num_pos_score_objects_act = sum(
            #[1 for x in scores_act.values() if x != 'nan' and x > 0.0])
            [1 for x in scores_act.values() if x > 0.0])
        num_neg_score_objects_act = sum(
            #[1 for x in scores_act.values() if x != 'nan' and x <= 0.0])
            [1 for x in scores_act.values() if x < 0.0])

        # ACT cluster level
        scores_clu_act = list()
        for g in ogs_act:
            if len(g) > 1:
                score_g = mean([scores_act[r] 
                    #for r in g if scores_act[r] != 'nan'])
                    for r in g if scores_act[r] != 0.0])
                max_score_g = amax([scores_act[r]
                    #for r in g if scores_act[r] != 'nan'])
                    for r in g if scores_act[r] != 0.0])
                scores_clu_act.append((score_g, max_score_g))
        num_pos_score_clusters_act = sum(
            [1 for x in scores_clu_act if x[0] > 0])
        num_neg_score_clusters_act = sum(
            [1 for x in scores_clu_act if x[0] <= 0])
        k_flag_act = all(
            [(x[1] >= mean_score_overall_act) for x in scores_clu_act])

        l_measured_values_act.append((
            mean_score_overall_act,
            variance_explained_percentage_act,
            num_singleton_clusters_act,
            num_pos_score_objects_act,
            num_neg_score_objects_act,
            num_pos_score_clusters_act,
            num_neg_score_clusters_act,
            k_flag_act
        ))

        # CASE object level
        mean_score_overall_case = mean(
            #[x for x in scores_case.values() if x != 'nan'])
            list(scores_case.values()))
        num_singleton_clusters_case = sum(
            #[1 for x in scores_case.values() if x == 'nan'])
            [1 for x in scores_case.values() if x == 0.0])
        num_pos_score_objects_case = sum(
            #[1 for x in scores_case.values() if x != 'nan' and x > 0.0])
            [1 for x in scores_case.values() if x > 0.0])
        num_neg_score_objects_case = sum(
            #[1 for x in scores_case.values() if x != 'nan' and x <= 0.0])
            [1 for x in scores_case.values() if x < 0.0])

        # CASE cluster level
        scores_clu_case = list()
        for g in ogs_case:
            if len(g) > 1:
                score_g = mean([scores_case[r]
                    #for r in g if scores_case[r] != 'nan'])
                    for r in g if scores_case[r] != 0.0])
                max_score_g = amax([scores_case[r]
                    #for r in g if scores_case[r] != 'nan'])
                    for r in g if scores_case[r] != 0.0])
                scores_clu_case.append((score_g, max_score_g))
        num_pos_score_clusters_case = sum(
            [1 for x in scores_clu_case if x[0] > 0])
        num_neg_score_clusters_case = sum(
            [1 for x in scores_clu_case if x[0] <= 0])
        k_flag_case = all(
            [(x[1] >= mean_score_overall_case) for x in scores_clu_case])

        l_measured_values_case.append((
            mean_score_overall_case,
            variance_explained_percentage_case,
            num_singleton_clusters_case,
            num_pos_score_objects_case,
            num_neg_score_objects_case,
            num_pos_score_clusters_case,
            num_neg_score_clusters_case,
            k_flag_case
        ))

        '''
        # case1 object level
        mean_score_overall_case1 = mean(
            #[x for x in scores_case1.values() if x != 'nan'])
            list(scores_case1.values()))
        num_singleton_clusters_case1 = sum(
            #[1 for x in scores_case1.values() if x == 'nan'])
            [1 for x in scores_case1.values() if x == 0.0])
        num_pos_score_objects_case1 = sum(
            #[1 for x in scores_case1.values() if x != 'nan' and x > 0.0])
            [1 for x in scores_case1.values() if x > 0.0])
        num_neg_score_objects_case1 = sum(
            #[1 for x in scores_case1.values() if x != 'nan' and x <= 0.0])
            [1 for x in scores_case1.values() if x < 0.0])

        # case1 cluster level
        scores_clu_case1 = list()
        for g in ogs_case1:
            if len(g) > 1:
                score_g = mean([scores_case1[r]
                    #for r in g if scores_case1[r] != 'nan'])
                    for r in g if scores_case1[r] != 0.0])
                max_score_g = amax([scores_case1[r]
                    #for r in g if scores_case1[r] != 'nan'])
                    for r in g if scores_case1[r] != 0.0])
                scores_clu_case1.append((score_g, max_score_g))
        num_pos_score_clusters_case1 = sum(
            [1 for x in scores_clu_case1 if x[0] > 0])
        num_neg_score_clusters_case1 = sum(
            [1 for x in scores_clu_case1 if x[0] <= 0])
        k_flag_case1 = all(
            [(x[1] >= mean_score_overall_case1) for x in scores_clu_case1])

        l_measured_values_case1.append((
            mean_score_overall_case1,
            variance_explained_percentage_case1,
            num_singleton_clusters_case1,
            num_pos_score_objects_case1,
            num_neg_score_objects_case1,
            num_pos_score_clusters_case1,
            num_neg_score_clusters_case1,
            k_flag_case1
        ))
        '''

    from csv import writer
    with open(fn_report, 'w+') as fout:
        writer = writer(fout)

        # values of 'k'
        writer.writerow(['VALUES of K'])
        writer.writerow(num_groups)
        writer.writerow(
            [l_measured_values_act[ik][-1] for ik, k in enumerate(num_groups)])
        writer.writerow(
            [l_measured_values_case[ik][-1] for ik, k in enumerate(num_groups)])
        #writer.writerow(
        #    [l_measured_values_case1[ik][-1] for ik, k in enumerate(num_groups)])
    
        for i, attribute in enumerate(items):
            writer.writerow([attribute])
            writer.writerow(l_measured_values_act[ik][i] 
                for ik, k in enumerate(num_groups))
            writer.writerow(l_measured_values_case[ik][i]
                for ik, k in enumerate(num_groups))
            #writer.writerow(l_measured_values_case1[ik][i]
            #    for ik, k in enumerate(num_groups))

