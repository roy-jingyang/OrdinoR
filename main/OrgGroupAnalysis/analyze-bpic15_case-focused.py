#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

from math import ceil
from pandas import DataFrame
from collections import defaultdict
from os.path import join

fn_event_log = sys.argv[1]
dirout = sys.argv[2]
fnout_resource_profiles = join(dirout, 'case-focused.profiles.csv')
fnout_group_profiles = join(dirout, 'case-focused.group-profiles.csv')
fnout_time_results = join(dirout, 'case-focused.time-avg.csv')
fnout_time_results1 = join(dirout, 'case-focused.time-full-info.csv')

if __name__ == '__main__':
    # Configuration based on given log (hard-coded)
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={
            'action_code': 4,
            '(case) last_phase': 5,
            'subprocess': 6,
            'phase': 7}) # bpic15-* (decoded)
        #el = read_disco_csv(f)

    # event log preprocessing
    from orgminer.Preprocessing.log_augmentation import append_case_duration
    el = append_case_duration(el)
    log = el

    from filters import filter_cases_by_frequency
    from filters import filter_events_by_frequency
    from filters import filter_events_by_active_resources

    # Case-level analysis
    # -------------------------
    # NOTE: convert the log to a case log (for case-level analysis)
    l_index = list()
    for case_id, events in log.groupby('case_id'):
        l_index.extend(events.drop_duplicates(subset=['resource']).index)
    log = log.loc[l_index]
    
    # NOTE: filter infrequent case classes (< 10%)
    log = filter_cases_by_frequency(log, '(case) last_phase', 0.1)
    # -------------------------

    # NOTE: Time-related analysis
    resource_case_timer = defaultdict(lambda: defaultdict(lambda: list()))
    for case_id, events in log.groupby('case_id'):
        case_class = set(events['(case) last_phase']).pop()
        case_duration = set(events['case_duration']).pop()

        for participant in set(events['resource']):
            resource_case_timer[participant][case_class].append(case_duration)

    # NOTE: filter resources with low involvement (< 1%)
    log = filter_events_by_active_resources(log, 0.01)


    from orgminer.ExecutionModeMiner.direct_groupby import (
        ATonlyMiner, CTonlyMiner)
    from orgminer.ResourceProfiler.raw_profiler import (
        count_execution_frequency)
    from orgminer.OrganizationalModelMiner.clustering.hierarchical import _ahc

    '''
        For clustering based approaches:
        each of the particular selection of "k" is evaluated by:
        - silhouette score
        - a value resulted from measuring by elbow method
    '''
    items = [
        'average_silhouette_score_overall',
        'value_elbow_method',
        'num_singletons',
        'num_res_pos_silhouette_score',
        'num_res_neg_silhouette_score',
    ]

    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from orgminer.Evaluation.m2m.cluster_validation import (
        variance_explained_percentage)
    from numpy import mean, amin, amax

    num_groups = list(range(2, len(set(log['resource']))))

    mode_miner = CTonlyMiner(log, case_attr_name='(case) last_phase')

    rl = mode_miner.derive_resource_log(log)

    #profiles = count_execution_frequency(rl)
    profiles = count_execution_frequency(rl, scale='normalize')

    # drop low-variance columns
    top_col_by_var = profiles.var(axis=0).sort_values(ascending=False)
    top_col_by_var = list(top_col_by_var.index)
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

        l_measured_values.append((
            mean_score_overall,
            var_pct,
            num_singleton_clusters,
            num_pos_score_objects,
            num_neg_score_objects,
            ))

    print('VALUES of K')
    print(','.join(str(k) for k in num_groups))
    print()
    for i, attribute in enumerate(items): 
        print(attribute)
        print(','.join(str(l_measured_values[ik][i])
            for ik, k in enumerate(num_groups)))
        print()

    print('Select a "K":')
    print('K = ', end='')
    next_k = int(input())

    ogs, _ = _ahc(profiles, next_k, method='ward')

    from numpy import set_printoptions, array
    set_printoptions(linewidth=100)
    
    resources = list(profiles.index)
    group_labels = array([-1] * len(resources))
    membership = dict()
    
    print('\nResource profiles in groups:')
    for i, og in enumerate(ogs):
        print('Group {}:'.format(i))
        print(profiles.loc[sorted(list(og))].values)
        for r in og:
            group_labels[resources.index(r)] = i # construct labels
            membership[r] = i

    # post-processing
    from copy import deepcopy
    df = deepcopy(profiles)
    df['Group label'] = group_labels
    df.sort_values(by='Group label', axis=0, inplace=True)

    # 1. resource profiles annotated with variance information
    var_row = profiles.var(axis=0)
    var_row.name = 'Variance'
    df.append(var_row).to_csv(fnout_resource_profiles)

    # 2. group profiles
    group_profiles_mat = defaultdict(lambda: defaultdict(lambda: 0))
    for resource, events in rl.groupby('resource'):
        group = membership[resource]
        for case_type, related_events in events.groupby('case_type'):
            group_profiles_mat[group][case_type] += len(related_events)
    l_group_size = list(len(ogs[group]) for group in group_profiles_mat.keys())
    df2 = DataFrame.from_dict(group_profiles_mat, orient='index').fillna(0)
    df2 = df2.div(df2.sum(axis=1), axis=0)
    df2 = df2[list(t[0] for t in df.columns if 'Group label' not in t)]
    df2['Group size'] = l_group_size
    print('\nGroup profiles:')
    print(df2)
    df2.to_csv(fnout_group_profiles)

    # 3. time performance analysis at resource-level (cont'd)
    resource_case_average_timer = defaultdict(
        lambda: defaultdict(lambda: None))

    from datetime import timedelta
    for case_class in set(log['(case) last_phase']):
        for resource in df.index:
            if len(resource_case_timer[resource][case_class]) > 0:
                avg_duration = timedelta(seconds=(
                    mean(resource_case_timer[resource][case_class])))
            else:
                avg_duration = ''
            resource_case_average_timer[resource][case_class] = str(
                avg_duration)

    df3 = DataFrame.from_dict(
        resource_case_average_timer, orient='index').fillna('nan')
    df3['Group label'] = df['Group label']
    df3.sort_values(by='Group label', axis=0, inplace=True)
    print('\nAverage cycle time of cases participated:')
    print(df3)
    df3.to_csv(fnout_time_results)

    # 4. original time performance information at resource-level
    rows = list()
    for case_id, events in log.groupby('case_id'):
        case_class = set(events['(case) last_phase']).pop()
        case_duration = set(events['case_duration']).pop()
        resources = set(events['resource'])
        for r in resources:
            rows.append(
                (membership[r], r, case_id, case_class, case_duration))
    from csv import writer
    with open(fnout_time_results1, 'w') as fout:
        writer = writer(fout)
        writer.writerow(
            ['Group', 'resource', 'case_id', 'permit_type', 'case_duration'])
        writer.writerows(rows)

