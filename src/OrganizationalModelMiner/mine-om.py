#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
from collections import defaultdict

f_event_log = sys.argv[1]
fout_org_model = sys.argv[2]
#mining_option = sys.argv[3]
#additional_params = sys.argv[4:] if len(sys.argv) > 4 else None

if __name__ == '__main__':
    # read event log as input
    cases = defaultdict(lambda: list())
    if f_event_log != 'none':
        with open(f_event_log, 'r', encoding='windows-1252') as f:
            is_header_line = True
            ln = 0
            # Exported from Disco:
            # BPiC 2013 Volvo VINST: Problem Mngt. Open problem
            # BPiC 2013 Volvo VINST: Problem Mngt. Closed problem
            # The 'WABO' event log data
            # BPiC 2012 Financial log
            for row in csv.reader(f):
                ln += 1
                if is_header_line:
                    is_header_line = False
                else:
                    caseid = row[0]
                    ctimestamp = row[3]
                    resource = row[2]
                    activity = row[1]
                    cases[caseid].append((caseid, activity, resource, ctimestamp))

        print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
        print('Average # of activities within each case: {}'.format(sum(
        len(x) for k, x in cases.items()) / len(cases.keys())))


    print('Input the number to choose a solution:')
    print('\t0. Default Mining (Song)')
    print('\t1. Metric based on Joint Activities (Song)')
    print('\t2. Agglomerative Hierarchical Clustering (Song)')
    print('\t3. Overlapping Community Detection (Appice)')
    print('\t4. Gaussian Mixture Model')
    print('\t5. Model based Overlapping Clustering')
    print('Option: ', end='')
    mining_option = int(input())

    if mining_option in [0, 1, 3]:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)
    elif mining_option == 2:
        from MiningOptions.Hierarchical import AHC
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        result = AHC.single_linkage(cases, num_groups)
    elif mining_option == 4:
        from MiningOptions.SoftClustering import GMM
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        result = GMM.mine(cases, num_groups)
    elif mining_option == 5:
        from MiningOptions.SoftClustering import MOC
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = input()
        result = MOC.mine(cases, num_groups)
    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    '''
    if mining_option.split('.')[0] == 'task':
        if mining_option.split('.')[1] == 'defaultmining':
            exit(1)
            # default mining requires original event log as the only input
            from MiningOptions import DefaultMining
            result = DefaultMining.mine(cases)
        elif mining_option.split('.')[1] == 'mja':
            exit(1)
            #threshold_value_step = float(additional_params[0])
            from MiningOptions.HardClustering import MJA
            result = MJA.threshold(cases)
            #result = MJA.threshold(cases, threshold_value_step)
        elif mining_option.split('.')[1] == 'community':
            exit(1)
            from MiningOptions.SoftClustering import Community
            f_sn_model = additional_params[0]
            result = Community.mine(f_sn_model)
        else:
            raise Exception('Option for task-based mining invalid.')
            exit(1)
    '''
    '''
    elif mining_option.split('.')[0] == 'case':
        raise Exception('Case-based mining under construction!')
        exit(1)
        from MiningOptions.HardClustering import MJC
        if mining_option.split('.')[1] == 'mjc_threshold':
            threshold_value = float(additional_params[0])
            result = MJC.threshold(cases, threshold_value)
        elif mining_option.split('.')[1] == 'mjc_remove':
            min_centrality = float(additional_params[0])
            if mining_option.split('.')[2] == 'degree':
                result = MJC.remove_by_degree(cases, min_centrality)
            elif mining_option.split('.')[2] == 'betweenness':
                result = MJC.remove_by_betweenness(cases, min_centrality)
            else:
                raise Exception('Option for case-based mining invalid.')
        else:
            raise Exception('Option for case-based mining invalid.')
    else:
        raise Exception('Failed to recognize input parameter!')
        exit(1)
    '''

    # try associating mined entities with tasks (entity assignment)
    assignments = defaultdict(lambda: set())
    for case_id, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            for entity_id, entity in result.items():
                if resource in entity:
                    assignments[entity_id].add(activity)
                else:
                    pass

    # output to file
    with open(fout_org_model, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_id', 'tasks', 'resources'])
        for entity_id in result.keys():
            writer.writerow([
                entity_id,
                ';'.join(t for t in assignments[entity_id]),
                ';'.join(r for r in result[entity_id])])

