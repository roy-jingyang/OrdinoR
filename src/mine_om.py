#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

fn_event_log = sys.argv[1]
fnout_org_model = sys.argv[2]


#mining_option = sys.argv[3]
#additional_params = sys.argv[4:] if len(sys.argv) > 4 else None

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    cases = read_disco_csv(fn_event_log)

    print('Input the number to choose a solution:')
    print('\t0. Default Mining (Song)')
    print('\t1. Metric based on Joint Activities (Song)')
    print('\t2. Agglomerative Hierarchical Clustering (Song)')
    print('\t3. Overlapping Community Detection (Appice)')
    print('\t4. Gaussian Mixture Model')
    print('\t5. Model based Overlapping Clustering')
    print('Option: ', end='')
    mining_option = int(input())

    from OrganizationalModelMiner.mining import *

    if mining_option in [3]:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)
    elif mining_option == 0:
        og = default_mining.mine(cases)

    elif mining_option == 2:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        og = hierarchical.AHC.single_linkage(cases)

    elif mining_option == 4:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        og = overlap.GMM.mine(cases)

    elif mining_option == 5:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        og = overlap.MOC.mine(cases)

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    '''
    if mining_option.split('.')[0] == 'task':
        elif mining_option.split('.')[1] == 'mja':
            exit(1)
            #threshold_value_step = float(additional_params[0])
            from mining.HardClustering import MJA
            og = MJA.threshold(cases)
            #og = MJA.threshold(cases, threshold_value_step)
        elif mining_option.split('.')[1] == 'community':
            exit(1)
            from mining.SoftClustering import Community
            f_sn_model = additional_params[0]
            og = Community.mine(f_sn_model)
    '''
    '''
    elif mining_option.split('.')[0] == 'case':
        raise Exception('Case-based mining under construction!')
        exit(1)
        from mining.HardClustering import MJC
        if mining_option.split('.')[1] == 'mjc_threshold':
            threshold_value = float(additional_params[0])
            og = MJC.threshold(cases, threshold_value)
        elif mining_option.split('.')[1] == 'mjc_remove':
            min_centrality = float(additional_params[0])
            if mining_option.split('.')[2] == 'degree':
                og = MJC.remove_by_degree(cases, min_centrality)
            elif mining_option.split('.')[2] == 'betweenness':
                og = MJC.remove_by_betweenness(cases, min_centrality)
            else:
                raise Exception('Option for case-based mining invalid.')
        else:
            raise Exception('Option for case-based mining invalid.')
    else:
        raise Exception('Failed to recognize input parameter!')
        exit(1)
    '''

    from OrganizationalModelMiner import entity_assignment
    assignment = entity_assignment.assign(og, cases)

    # save the mining og to a file
    from IO.writer import write_om_csv
    write_om_csv(fnout_org_model, og, assignment)
    #from IO.writer import write_om_omml

