#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

from math import ceil
from re import search as regex_search
from pandas import read_csv

fn_event_log = sys.argv[1]
fnout = sys.argv[2]

def filter_cases_by_frequency(log, case_attr, threshold=0.1):
    frequent_case_classes = set()
    num_total_cases = len(set(log['case_id']))

    for case_class, events in log.groupby(case_attr):
        num_cases = len(set(events['case_id']))
        if (num_cases / num_total_cases) >= threshold:
            frequent_case_classes.add(case_class)
    
    filtered = log.loc[log[case_attr].isin(frequent_case_classes)]
    print('''[filter_cases_by_frequency] 
        Frequent values of attribute "{}": {}
        {} / {} ({:.1%}) cases are kept.'''.format(
        case_attr, set(frequent_case_classes),
        len(set(filtered['case_id'])),
        num_total_cases,
        len(set(filtered['case_id'])) / num_total_cases))
    return filtered

def filter_events_by_frequency(log, event_attr, threshold=0.1):
    frequent_classes = set()
    num_total_events = len(log)

    for class_name, events in log.groupby(event_attr):
        if (len(events) / num_total_events) >= threshold:
            frequent_classes.add(class_name)

    filtered = log.loc[log[event_attr].isin(frequent_classes)]
    print('''[filter_events_by_frequency] 
        Frequent values of attribute "{}": {}
        {} / {} ({:.1%}) events are kept.'''.format(
        event_attr, set(frequent_classes),
        len(filtered),
        num_total_events,
        len(filtered) / num_total_events))
    return filtered

def filter_events_by_active_resources(log, threshold=0.01):
    active_resources = set()
    num_total_events = len(log)

    for resource, events in log.groupby('resource'):
        if (len(events) / num_total_events) >= threshold and resource != '':
            active_resources.add(resource)

    filtered = log.loc[log['resource'].isin(active_resources)]
    print('''{} resources found active with {} events.\n'''.format(
        len(set(filtered['resource'])), len(filtered)))
    return filtered

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)

    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    # TODO
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={
            'action_code': 4,
            '(case) last_phase': 5,
            'subprocess': 6,
            'phase': 7}) # bpic15-* (decoded)
        #el = read_disco_csv(f)

    # event log preprocessing
    log = el
    """
    print('''Select the desired data granularity for analysis:\n
        [1] Event-level\n
        [2] Case-level\n
    Input a number: ''', end='')
    OPT_granularity = input()
    if OPT_granularity == '1':
        convert_to_cases = False
    elif OPT_granularity == '2':
        convert_to_cases = True
    else:
        raise ValueError("Unrecognized selection.")
    """

    # NOTE: Case-level analysis
    # -------------------------
    '''
    # NOTE: convert the log to a case log (keep only 1 event for each case)
    # (for case-level analysis)
    l_index = list()
    for case_id, events in log.groupby('case_id'):
        l_index.extend(events.drop_duplicates(subset=['resource']).index)
    log = log.loc[l_index]

    # NOTE: filter infrequent case classes (< 10%)
    log = filter_cases_by_frequency(log, '(case) last_phase', 0.1)
    print('Select a case class to perform analysis: ', end='')
    selected_case_class = input()
    if selected_case_class == '':
        print('\tOmitted')
    else:
        frequent_case_classes = {selected_case_class}
        log = log.loc[log['(case) last_phase'].isin(frequent_case_classes)]
    '''
    # -------------------------

    # NOTE: Event-level analysis
    # -------------------------
    # NOTE: filter infrequent case classes (< 10%)
    log = filter_cases_by_frequency(log, '(case) last_phase', 0.1)
    print('Select a case class to perform analysis: ', end='')
    selected_case_class = input()
    if selected_case_class == '':
        print('\tOmitted')
    else:
        frequent_case_classes = {selected_case_class}
        log = log.loc[log['(case) last_phase'].isin(frequent_case_classes)]

    # NOTE: filter infrequent subprocesses (< 10%)
    log = filter_events_by_frequency(log, 'subprocess', 0.1)

    # NOTE: filter infrequent activity classes, "phase" (< 10%)
    log = filter_events_by_frequency(log, 'phase', 0.1)
    print('Select an activity class to perform analysis: ', end='')
    selected_activity_class = input()
    if selected_activity_class == '':
        print('\tOmitted')
    else:
        frequent_activity_classes = {selected_activity_class}
        log = log.loc[log['phase'].isin(frequent_activity_classes)]

    log['activity'] = log['phase']
    # -------------------------

    # NOTE: filter resources with low involvement (< 1%)
    log = filter_events_by_active_resources(log, 0.01)


    from orgminer.ExecutionModeMiner.direct_groupby import (
        ATonlyMiner, CTonlyMiner, ATCTMiner)
    from orgminer.ResourceProfiler.raw_profiler import (
        count_execution_frequency)
    from orgminer.OrganizationalModelMiner.clustering.hierarchical import (
        _ahc)

    '''
        For clustering based approaches:
        each of the particular selection of "k" is evaluated by both:
        - a value resulted from measuring by elbow method
        - decision drawn from silhouette analysis
    '''
    ''' {8} items may be collected for analysis:
        - the result from silhouette analysis (k_flag),
        - the average value of silhouette score (of all resources), excluding
          those designated to singleton clusters;
        - the value from elbow method (of the whole clustering);
        - the number of singleton clusters;
        - the number of resources with POSITIVE silhouette score;
        - the number of resources with NEGATIVE silhouette score;
        - the number of clusters with POS average silhouette score (excluding
          singletons);
        - the number of clusters with NEG average silhouette score (excluding
          singletons);
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
    ]
    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from orgminer.Evaluation.m2m.cluster_validation import (
        variance_explained_percentage)
    from numpy import mean, amin, amax

    num_groups = list(range(2, len(set(log['resource']))))

    mode_miner = ATonlyMiner(log)
    #mode_miner = CTonlyMiner(log, case_attr_name='(case) last_phase')
    #mode_miner = ATCTMiner(log, case_attr_name='(case) last_phase')

    iteration = 1
    MAX_ITER = 2
    discarded_resources_singleton = list()
    discarded_resources_negative = list()
    while iteration <= MAX_ITER:
        print('-' * 30 + 'Iteration: {}'.format(iteration) + '-' * 30)
        rl = mode_miner.derive_resource_log(log)

        #profiles = count_execution_frequency(rl)
        profiles = count_execution_frequency(rl, scale='normalize')

        # drop the "unknowns" (bpic15-*)
        if 'AT.' in profiles.columns:
            profiles.drop(columns='AT.', inplace=True)

        # drop low-variance columns
        top_col_by_var = profiles.var(axis=0).sort_values(ascending=False)
        top_col_by_var = list(
            #top_col_by_var[:ceil(len(profiles.columns)*0.5)].index)
            #top_col_by_var[:min(10, len(profiles.columns))].index)
            top_col_by_var[:].index)
        profiles = profiles[top_col_by_var]

        print('{} columns remain in the profiles'.format(
            len(profiles.columns)))

        l_measured_values = list()
        for k in num_groups:
            # calculate silhouette scores and variance explained (for each K)
            ogs, _ = _ahc(profiles, k, method='ward')

            scores = silhouette_score(ogs, profiles)
            var_pct = variance_explained_percentage(
                ogs, profiles)

            # object-level
            mean_score_overall = mean(
                list(scores.values()))
            num_pos_score_objects = sum(
                [1 for x in scores.values() if x > 0.0])
            num_neg_score_objects = sum(
                [1 for x in scores.values() if x < 0.0])

            # cluster-level
            num_singleton_clusters = sum(
                [1 for og in ogs if len(og) == 1])
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

            l_measured_values.append((
                k_flag,
                mean_score_overall,
                var_pct,
                num_singleton_clusters,
                num_pos_score_objects,
                num_neg_score_objects,
                num_pos_score_clusters,
                num_neg_score_clusters,
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
            print('\tfor proceeding to the next iteration, type in the '
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

            ogs, _ = _ahc(profiles, next_k, method='ward')

            if stop_iteration:
                from numpy import set_printoptions, array
                set_printoptions(linewidth=100)
                
                resources = list(profiles.index)
                group_labels = array([-1] * len(resources))
                for i, og in enumerate(ogs):
                    print('Group {}:'.format(i))
                    print(profiles.loc[sorted(list(og))].values)
                    for r in og:
                        group_labels[resources.index(r)] = i # construct labels
                iteration = MAX_ITER

                # NOTE: for export use
                from copy import deepcopy
                df = deepcopy(profiles)
                df['Group label'] = group_labels
                var_row = profiles.var(axis=0)
                var_row.name = 'Variance'
                df.append(var_row).to_csv(fnout)
                exit()

            else:
                exit()
                '''
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
                '''

        else:
            print('\nMAX iteration reached.')

        # proceed to the next iteration
        iteration += 1

