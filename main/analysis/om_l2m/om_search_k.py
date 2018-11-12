#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout = sys.argv[2]

def search_k(profiles, num_groups, method):
    if method == 'mja':
        from OrganizationalModelMiner.disjoint.graph_partitioning import mja
        return mja(profiles, num_groups, metric='correlation',
            search_only=True)
    elif method == 'gmm':
        from OrganizationalModelMiner.overlap.clustering import gmm
        #return gmm(profiles, num_groups, threshold=None, init='ahc',
        return gmm(profiles, num_groups, threshold=None,
            search_only=True)
    elif method == 'moc':
        from OrganizationalModelMiner.overlap.clustering import moc
        #return moc(profiles, num_groups, init='ahc',
        return moc(profiles, num_groups,
            search_only=True)
    elif method == 'fcm':
        from OrganizationalModelMiner.overlap.clustering import fcm
        #return fcm(profiles, num_groups, threshold=None, init='ahc',
        return fcm(profiles, num_groups, threshold=None,
            search_only=True)
    else:
        exit('[Error] Unrecognized method option')

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    print('Input the desired range [low, high): ', end=' ')
    num_groups = input()
    num_groups = num_groups[1:-1].split(',')
    num_groups = range(int(num_groups[0]), int(num_groups[1]))

    # build profiles
    from ResourceProfiler.raw_profiler import performer_activity_frequency
    profiles = performer_activity_frequency(rl, use_log_scale=False)

    methods = ['gmm', 'moc', 'fcm', 'mja']
    from multiprocessing import Pool
    from functools import partial
    partial_search_k = partial(search_k,
            profiles, num_groups)
    with Pool(len(methods)) as p:
        best_ks = p.map(partial_search_k, methods)

    with open(fnout, 'w') as f:
        f.write('-' * 35 + 'Best "K"' + '-' * 35 + '\n')
        for method, best_k in zip(methods, best_ks):
            f.write('{}\t{}\n'.format(method.upper(), best_k))

