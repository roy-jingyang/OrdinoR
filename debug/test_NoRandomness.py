#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# List input parameters from shell
fn_event_log = sys.argv[1]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    from ResourceProfiler.raw_profiler import performer_activity_frequency
    profiles = performer_activity_frequency(rl, use_log_scale=False)

    # specify the number of tests to be performed to check consistency
    n_tests = 10
    #from OrganizationalModelMiner.disjoint.graph_partitioning import mja
    #from OrganizationalModelMiner.hierarchical.clustering import ahc
    from OrganizationalModelMiner.overlap.community_detection import link_partitioning
    from OrganizationalModelMiner.overlap.clustering import gmm
    from OrganizationalModelMiner.overlap.clustering import moc
    from OrganizationalModelMiner.overlap.clustering import fcm

    prev_ogs = None
    succ = True
    for t in range(n_tests):
        #ogs = mja(profiles, range(5, 6), metric='correlation')
        #ogs, _ = ahc(profiles, range(5, 6), method='ward')
        ogs = link_partitioning(profiles, metric='correlation')
        #ogs = gmm(profiles, range(5, 6), init='ahc', threshold=None)
        #ogs = moc(profiles, range(5, 6), init='ahc')
        #ogs = fcm(profiles, range(5, 6), init='ahc', threshold=None)
        
        if t != 0:
            # compare
            if sorted(ogs) == sorted(prev_ogs):
                pass
            else:
                print('Inconsistent results found!')
                print('\tCurrent:')
                print(len(sorted(ogs)))
                print('\tPrevious:')
                print(len(sorted(prev_ogs)))
                succ = False
                break
        prev_ogs = ogs

    if succ:
        print('\n{} tests passed.'.format(n_tests))
        for i, og in enumerate(prev_ogs):
            print(i)
            print(og)

