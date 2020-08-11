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
    from orgminer.IO.reader import read_xes
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_xes(f)

    from orgminer.ExecutionModeMiner import direct_groupby
    #exec_mode_miner = direct_groupby.ATonlyMiner(el)
    exec_mode_miner = direct_groupby.FullMiner(el, 
        case_attr_name='(case)_channel', resolution='weekday')

    rl = exec_mode_miner.derive_resource_log(el)

    # build profiles
    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl, scale=None)
    '''
    from orgminer.ResourceProfiler.pattern_profiler import \
        from_frequent_patterns
    profiles = from_frequent_patterns(rl)
    '''

    '''
    from sklearn.decomposition import PCA
    from pandas import DataFrame
    profiles = DataFrame(
        PCA(n_components=None).fit_transform(profiles.to_numpy()),
        index=profiles.index
    )
    print(profiles)
    '''

    num_groups = [10]
    from orgminer.OrganizationalModelMiner.clustering.hierarchical import ahc
    from orgminer.OrganizationalModelMiner.clustering.overlap import moc
    ogs = ahc(profiles, num_groups, method='ward')
    #ogs = moc(profiles, num_groups, init='kmeans')
    #ogs = ahc(profiles, num_groups, method='average', metric='jaccard')

    # 3. Assign execution modes to groups
    from orgminer.OrganizationalModelMiner.group_profiling import \
        full_recall, overall_score, association_rules

    #om = full_recall(ogs, rl)
    #om = overall_score(ogs, rl, w1=0.5, p=0.5)
    om = association_rules(ogs, rl)

    '''
    from orgminer.OrganizationalModelMiner.base import OrganizationalModel
    from numpy import nonzero
    om = OrganizationalModel()
    for group in ogs:
        modes = set()
        for r in group:
            r_modes = set(profiles.columns[
                list(nonzero(profiles.loc[r].to_numpy())[0])
            ])
            modes.update(r_modes)
        om.add_group(group, sorted(list(modes)))
    '''

    # Evaluate discovery result

    print('-' * 80)
    measure_values = list()
    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from numpy import mean
    silhouette_score = mean(list(silhouette_score(
        ogs, profiles, metric='euclidean'
    ).values()))
    print('Silhouette\t= {:.6f}'.format(silhouette_score))
    print('-' * 80)
    print()
    
    from orgminer.Evaluation.l2m import conformance
    fitness_score = conformance.fitness(rl, om)
    print('Fitness\t\t= {:.3f}'.format(fitness_score))
    measure_values.append(fitness_score)
    print()
    precision_score = conformance.precision(rl, om)
    print('Precision\t= {:.3f}'.format(precision_score))
    measure_values.append(precision_score)
    print()

