#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import networkx as nx

f_event_log = sys.argv[1]
f_social_network = sys.argv[2]
fout_org_model = sys.argv[3]
mining_option = sys.argv[4]
additional_params = sys.argv**

if __name__ == '__main__':
    # read event log as input
    cases = defaultdict(lambda: list())
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

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))

    # read social network (if provided) as input
    if f_social_network == 'none':
        g = None
    else:
        g = nx.read_gml(f_social_network)
    
    # try mining organizational entities
    try:
        if mining_option.split('.')[0] == 'task':
            if mining_option.split('.')[1] == 'defaultmining':
                # default mining requires original event log as the only input
                from MiningOptions import DefaultMining
                result = DefaultMining.mine(cases)
            elif mining_option.split('.').[1] == 'mja':
                from MiningOptions import MJA
                threshold_value = float(additional_params[5])
                result = MJA.threshold(g, threshold_value)
            elif mining_option.split('.')[1] == 'AHC':
                from MiningOptions import AHC
                k_clusters = int(additional_params[5])
                result = AHC.cluster(g, k_clusters)
            else:
                exit(1)
        elif mining_option.split('.')[0] == 'case':
            from MiningOptions import MJC
            if mining_option.split('.')[1] == 'mjc_threshold':
                threshold_value = float(additional_params[5])
                result = MJC.threshold(g, threshold_value)
            else mining_option.split('.')[1] == 'mjc_remove':
                min_centrality = float(additional_params[5])
                if mining_option.split('.')[2] == 'degree':
                    result = MJC.remove_by_degree(g, min_centrality)
                elif mining_option.split('.')[2] == 'betweenness':
                    result = MJC.remove_by_betweenness(g, min_centrality)
                else:
                    exit(1)
            else:
                exit(1)
        else:
            exit(1)
    except Exception as e:
        print(e)

    # try associating mined entities with tasks (entity assignment)
    assignments = defaultdict(lambda: set())
    for case_id, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][3]
            for entity_id, entity in result.items():
                if resource in entity:
                    assignment[entity_id].add(activity)
                else:
                    pass

    # output to file
    with open(fout_org_model, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_id', 'tasks', 'resources'])
        for entity_id in result.keys():
            writer.writerow([
                entity_id,
                ';'.join(t for t in assignment[entity_id]),
                ';'.join(r for r in result[entity_id])])

