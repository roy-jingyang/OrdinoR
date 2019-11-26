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
fnout_resource_profiles = join(dirout, 'act-focused.profiles.csv')
fnout_group_profiles = join(dirout, 'act-focused.group-profiles.csv')
fnout_time_results = join(dirout, 'act-focused.time-avg.csv')
fnout_time_results1 = join(dirout, 'act-focused.time-full-info.csv')

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
    log = el

    from filters import filter_cases_by_frequency
    from filters import filter_events_by_frequency
    from filters import filter_events_by_active_resources

    # Event-level analysis
    # -------------------------
    # NOTE: filter infrequent subprocesses (< 10%)
    log = filter_events_by_frequency(log, 'subprocess', 0.1)
    print('Select a subprocess to perform analysis: ', end='')
    selected_subprocess = input()
    if selected_subprocess == '':
        print('\t(Omitted)')
    else:
        frequent_activity_classes = {selected_activity_class}
        log = log.loc[log['subprocess'] == selected_subprocess]

    # NOTE: filter infrequent activity classes, "phase" (< 10%)
    log = filter_events_by_frequency(log, 'phase', 0.1)
    log['activity'] = log['phase']
    # -------------------------

    # NOTE: Time-related analysis
    resource_phase_timer = defaultdict(lambda: defaultdict(lambda: list()))
    from datetime import datetime
    for case_id, trace in log.groupby('case_id'):
        for phase, events in trace.groupby('phase'):
            start_time = datetime.strptime(
                events.iloc[0]['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
            end_time = datetime.strptime(
                events.iloc[-1]['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
            resources = set(events['resource'])
            phase_duration = (end_time - start_time).total_seconds()
            for r in resources:
                resource_phase_timer[r][phase].append(phase_duration)

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

    mode_miner = ATonlyMiner(log)

    rl = mode_miner.derive_resource_log(log)

    #profiles = count_execution_frequency(rl)
    profiles = count_execution_frequency(rl, scale='normalize')

    # sort columns based on variance
    top_col_by_var = profiles.var(axis=0).sort_values(ascending=False)
    top_col_by_var = list(top_col_by_var.index)
    profiles = profiles[top_col_by_var]

    print('{} columns remain in the profiles'.format(len(profiles.columns)))

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
    from collections import defaultdict
    group_profiles_mat = defaultdict(lambda: defaultdict(lambda: 0))
    for resource, events in rl.groupby('resource'):
        group = membership[resource]
        for act_type, related_events in events.groupby('activity_type'):
            group_profiles_mat[group][act_type] += len(related_events)
    l_group_size = list(len(ogs[group]) for group in group_profiles_mat.keys())
    df2 = DataFrame.from_dict(group_profiles_mat, orient='index').fillna(0)
    df2 = df2.div(df2.sum(axis=1), axis=0)
    df2 = df2[list(t[1] for t in df.columns if 'Group label' not in t)]
    df2['Group size'] = l_group_size
    print('\nGroup profiles:')
    print(df2)
    df2.to_csv(fnout_group_profiles)

    # 3. time performance (on average) at resource-level (cont'd)
    resource_phase_average_timer = defaultdict(
        lambda: defaultdict(lambda: None))

    from datetime import timedelta
    for phase in set(log['phase']):
        for resource in df.index:
            if len(resource_phase_timer[resource][phase]) > 0:
                avg_duration = timedelta(seconds=(
                    mean(resource_phase_timer[resource][phase])))
            else:
                avg_duration = ''
            resource_phase_average_timer[resource][phase] = str(
                avg_duration)

    df3 = DataFrame.from_dict(
        resource_phase_average_timer, orient='index').fillna('nan')
    df3['Group label'] = df['Group label']
    print('\nAverage cycle time of phases participated:')
    print(df3)
    df3.to_csv(fnout_time_results)

    # 4. original time performance information at resource-level
    rows = list()
    for case_id, trace in log.groupby('case_id'):
        for phase, events in trace.groupby('phase'):
            start_time = datetime.strptime(
                events.iloc[0]['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
            end_time = datetime.strptime(
                events.iloc[-1]['timestamp'], '%Y/%m/%d %H:%M:%S.%f')
            resources = set(events['resource'])
            phase_duration = (end_time - start_time).total_seconds()
            for r in resources:
                rows.append(
                    (membership[r], r, case_id, phase, phase_duration))

    from csv import writer
    with open(fnout_time_results1, 'w') as fout:
        writer = writer(fout)
        writer.writerow(
            ['Group', 'resource', 'case_id', 'phase', 'phase_duration'])
        writer.writerows(rows)

