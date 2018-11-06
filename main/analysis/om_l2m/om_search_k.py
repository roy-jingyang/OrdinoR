#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
    num_groups = input()
    num_groups = num_groups[1:-1].split(',')
    num_groups = range(int(num_groups[0]), int(num_groups[1]))

    # build profiles
    from ResourceProfiler.raw_profiler import performer_activity_frequency
    profiles = performer_activity_frequency(rl, use_log_scale=False)

    from OrganizationalModelMiner.disjoint.graph_partitioning import mja
    best_k_mja = mja(profiles, num_groups, metric='correlation',
            search_only=True)

    from OrganizationalModelMiner.hierarchical import clustering
    best_k_ahc = clustering.ahc(profiles, num_groups, method='ward',
            search_only=True)

    from OrganizationalModelMiner.overlap.clustering import gmm, moc, fcm
    best_k_gmm = gmm(profiles, num_groups, threshold=None, init='ahc',
            search_only=True)
    best_k_moc = moc(profiles, num_groups, init='ahc',
            search_only=True)
    best_k_fcm = fcm(profiles, num_groups, threshold=None, init='ahc',
            search_only=True)

    with open(fnout, 'w') as f:
        f.write('-' * 35 + 'Best "K"' + '-' * 32 + '\n')
        f.write('MJA\t{}\n'.format(best_k_mja))
        f.write('AHC\t{}\n'.format(best_k_ahc))
        f.write('GMM\t{}\n'.format(best_k_gmm))
        f.write('MOC\t{}\n'.format(best_k_moc))
        f.write('FCM\t{}\n'.format(best_k_fcm))

