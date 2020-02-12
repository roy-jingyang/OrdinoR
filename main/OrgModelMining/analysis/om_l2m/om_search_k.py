#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fnout = sys.argv[2]

def search_k(profiles, num_groups, method):
    if method == 'mja':
        from orgminer.OrganizationalModelMiner.community.graph_partitioning \
            import mja
        return mja(profiles, num_groups, metric='correlation',
            search_only=True)
    elif method == 'ahc':
        from orgminer.OrganizationalModelMiner.clustering.hierarchical \
            import ahc
        return ahc(profiles, num_groups, method='ward', metric='euclidean',
            search_only=True)
    elif method == 'gmm':
        from orgminer.OrganizationalModelMiner.clustering.overlap import gmm
        return gmm(profiles, num_groups, threshold=None, init='kmeans',
            search_only=True)
    elif method == 'moc':
        from orgminer.OrganizationalModelMiner.clustering.overlap import moc
        return moc(profiles, num_groups, init='kmeans',
            search_only=True)
    elif method == 'fcm':
        from orgminer.OrganizationalModelMiner.clustering.overlap import fcm
        return fcm(profiles, num_groups, threshold=None, init='kmeans',
            search_only=True)
    else:
        exit('[Error] Unrecognized method option')

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
    from orgminer.ExecutionModeMiner.informed_groupby import \
        TraceClusteringFullMiner

    #mode_miner = ATonlyMiner(el)
    mode_miner = FullMiner(el, 
        case_attr_name='(case) LoanGoal', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/bpic12.bosek5.tcreport',
    #    resolution='weekday')

    rl = mode_miner.derive_resource_log(el)

    print('Input the desired range [low, high): ', end=' ')
    num_groups = input()
    num_groups = num_groups[1:-1].split(',')
    num_groups = list(range(int(num_groups[0]), int(num_groups[1])))

    # build profiles
    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl)

    methods = ['ahc', 'moc']
    from multiprocessing import Pool
    from functools import partial
    partial_search_k = partial(search_k, profiles, num_groups)
    best_ks = list(map(partial_search_k, methods))

    with open(fnout, 'w') as f:
        f.write('-' * 35 + 'Best "K"' + '-' * 35 + '\n')
        for method, best_k in zip(methods, best_ks):
            f.write('{}\t{}\n'.format(method.upper(), best_k))

