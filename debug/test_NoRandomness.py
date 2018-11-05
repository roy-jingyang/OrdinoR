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
    n_tests = 100
    #from OrganizationalModelMiner.disjoint.graph_partitioning import mja as miner
    #from OrganizationalModelMiner.hierarchical.clustering import ahc as miner
    #from OrganizationalModelMiner.overlap.community_detection import link_partitioning as miner
    #from OrganizationalModelMiner.overlap.clustering import gmm as miner
    #from OrganizationalModelMiner.overlap.clustering import moc as miner
    from OrganizationalModelMiner.overlap.clustering import fcm as miner

    prev_ogs = None
    succ = True
    for t in range(n_tests):
        #ogs = miner(profiles, range(5, 6), metric='correlation')
        #ogs, _ = miner(profiles, range(5, 6), method='ward')
        #ogs = miner(profiles, metric='correlation')
        # TODO: GMM didn't pass.
        #ogs = miner(profiles, range(5, 6), init='zero', threshold=None)
        # TODO: MOC didn't pass
        #ogs = miner(profiles, range(5, 6), init='zero')
        # TODO: FCM didn't pass.
        ogs = miner(profiles, range(5, 6), init='zero', threshold=None)

        if t != 0:
            # compare
            if sorted(ogs) == sorted(prev_ogs):
                pass
            else:
                print('Inconsistent results found!')
                print('\tCurrent:')
                print(sorted(ogs))
                print('\tPrevious:')
                print(sorted(prev_ogs))
                succ = False
                break
        prev_ogs = ogs

    if succ:
        print('\n{} tests passed.'.format(n_tests))
        for og in prev_ogs:
            print(og)

