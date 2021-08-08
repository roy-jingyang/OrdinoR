#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('fn_event_log', 
    help='Path to input log file')

args = parser.parse_args()

fn_event_log = args.fn_event_log

if __name__ == '__main__':
    # read event log as input
    from ordinor.io import read_xes
    el = read_xes(fn_event_log)

    print(el)

    from ordinor.execution_context import ATonlyMiner, FullMiner
    exe_ctx_miner = ATonlyMiner(el)
    #exe_ctx_miner = FullMiner(el, 
    #    case_attr_name='(case)_channel', resolution='weekday'
    #)

    rl = exe_ctx_miner.derive_resource_log(el)

    # build profiles
    from ordinor.org_model_miner.resource_features import direct_count
    profiles = direct_count(rl, scale=None)

    num_groups = [10]
    from ordinor.org_model_miner.group_discovery import ahc, moc
    #ogs = ahc(profiles, num_groups, method='ward')
    ogs = moc(profiles, num_groups, init='kmeans')

    # Evaluate grouping discovery
    print('-' * 80)
    measure_values = list()
    from ordinor.analysis.m2m import silhouette_score
    from numpy import mean
    silhouette_score = mean(list(silhouette_score(
        ogs, profiles, metric='euclidean'
    ).values()))
    print('Silhouette\t= {:.6f}'.format(silhouette_score))
    print('-' * 80)
    print()

    l_group_size = sorted(
        [(i, len(og)) for i, og in enumerate(ogs)],
        key=lambda x: x[1], reverse=True
    )
    for (i, length) in l_group_size:
        print('Group {};{}'.format(i, length))
    print()

    # 3. Assign execution contexts to groups
    from ordinor.org_model_miner.group_profiling import \
        full_recall, overall_score

    #om = full_recall(ogs, rl)
    om = overall_score(ogs, rl, w1=0.5, p=0.5)

    # Evaluate final discovery result
    from ordinor.conformance import fitness, precision
    fitness_score = fitness(rl, om)
    print('Fitness\t\t= {:.3f}'.format(fitness_score))
    measure_values.append(fitness_score)
    print()
    precision_score = precision(rl, om)
    print('Precision\t= {:.3f}'.format(precision_score))
    measure_values.append(precision_score)
    print()

