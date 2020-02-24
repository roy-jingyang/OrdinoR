#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
sys.path.append('../')

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
    exec_mode_miner = direct_groupby.ATonlyMiner(el)

    rl = exec_mode_miner.derive_resource_log(el)


    # build profiles
    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl)
    num_groups = 9
    from orgminer.OrganizationalModelMiner.clustering.overlap import moc
    ogs = moc(profiles, num_groups,
        init='kmeans')

    # 3. Assign execution modes to groups
    from orgminer.OrganizationalModelMiner.mode_assignment import \
        full_recall, overall_score

    #om = overall_score(ogs, rl, auto_search=True)
    om = overall_score(ogs, rl, w1=0.2, p=0.7)

    # Evaluate discovery result

    print('-' * 80)
    measure_values = list()
    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from numpy import mean
    silhouette_score = mean(list(silhouette_score(ogs, profiles).values()))
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


