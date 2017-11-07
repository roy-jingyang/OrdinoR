#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import networkx as nx
from collections import defaultdict

f_event_log = sys.argv[1]
fout_org_model = sys.argv[2]
mining_option = sys.argv[3]
additional_params = sys.argv[4:] if len(sys.argv) > 4 else None

if __name__ == '__main__':
    # read event log as input
    cases = defaultdict(lambda: list())
    resources = set()
    with open(f_event_log, 'r', encoding='windows-1252') as f:
        is_header_line = True
        ln = 0
        '''
        # BPiC 2013 Volvo Service Desk: Incident Mngt. Syst.
        for line in f:
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                row = line.split(';')
                caseid = row[0] # SR Number
                ctimestamp = row[1] # Change Date+Time
                resource = row[-1]
                activity = row[2] + row[3]
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        # BPiC 2015 Building Permit Application: Municiality 3
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0] 
                ctimestamp = row[3] # Complete timestamp 
                resource = row[2]
                activity = row[1] # Activity code
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        '''
        # The 'WABO' event log data
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
                resources.add(resource)

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))

    
    # try mining organizational entities
    if mining_option.split('.')[0] == 'task':
        if mining_option.split('.')[1] == 'defaultmining':
            # default mining requires original event log as the only input
            from MiningOptions import DefaultMining
            result = DefaultMining.mine(cases)
        elif mining_option.split('.')[1] == 'mja':
            threshold_value_step = float(additional_params[0])
            from MiningOptions.HardClustering import MJA
            result = MJA.threshold(cases, threshold_value_step)
        elif mining_option.split('.')[1] == 'ahc':
            raise Exception('Hierarchical mining under construction!')
            #TODO
            '''
            k_cluster_step = 1
            from MiningOptions.Hierarchical import AHC
            result = AHC.single_linkage(cases, k_cluster_step)
            '''
        elif mining_option.split('.')[1] == 'gmm':
            #threshold_value = float(additional_params[0])
            from MiningOptions.SoftClustering import GMM
            #result = GMM.mine(cases, threshold_value)
            result = GMM.mine(cases)
        elif mining_option.split('.')[1] == 'bgm':
            #threshold_value_step = float(additional_params[0])
            from MiningOptions.SoftClustering import BGM
            #result = BGM.mine(cases, threshold_value_step)
            result = BGM.mine(cases)
        else:
            raise Exception('Option for task-based mining invalid.')
    elif mining_option.split('.')[0] == 'case':
        raise Exception('Case-based mining under construction!')
        '''
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
        '''
    else:
        raise Exception('Failed to recognize input parameter!')

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

