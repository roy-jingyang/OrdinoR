#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# List input parameters from shell
fn_event_log = sys.argv[1]

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
    #naive_exec_mode_miner = ATonlyMiner(el)
    naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')
    rl = naive_exec_mode_miner.derive_resource_log(el)

    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl, scale='log')

    # specify the number of tests to be performed to check consistency
    n_tests = 10
    '''
    from orgminer.OrganizationalModelMiner.community.overlap import link_partitioning

    prev_ogs = None
    succ = True
    for t in range(n_tests):
        ogs = link_partitioning(profiles, n_groups=10, metric='correlation')
        
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
    '''

