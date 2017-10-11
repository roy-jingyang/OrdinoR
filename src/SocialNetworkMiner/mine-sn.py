#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx

f_event_log = sys.argv[1]
fout_social_network = sys.argv[2]
mining_option = sys.argv[3]

if __name__ == '__main__':
    cases = defaultdict(lambda: list())
    with open(f_event_log, 'r', encoding='windows-1252') as f:
        is_header_line = True
        ln = 0
        # BPiC 2013 Volvo Service Desk: Incident Mngt. Syst.
        '''
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
                cases[caseid].append((caseid, ctimestamp, resource, activity))
        '''
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
                cases[caseid].append((caseid, ctimestamp, resource, activity))

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))

    try:
        if mining_option.split('.')[0] == 'pc':
            from MiningOptions import PossibleCausality
            if mining_option.split('.')[1] == 'handover_CDCM':
                # Our main concern now
                # handover.ICCDCM(cases=list_of_cases, is_task_specific=false)
                result = PossibleCausality.handover_CDCM(cases)
            elif mining_option.split('.')[1] == 'handover_duration':
                # Our main concern now
                result = PossibleCausality.handover_duration(cases)
            elif mining_option.split('.')[1] == 'handover_CDIM':
                result = PossibleCausality.handover_CDIM(cases)
            elif mining_option.split('.')[1] == 'handover_CICM':
                result = PossibleCausality.handover_CICM(cases)
            elif mining_option.split('.')[1] == 'handover_CIIM':
                result = PossibleCausality.handover_CIIM(cases)
            else:
                exit(1)
        elif mining_option.split('.')[0] == 'mjc':
            from MiningOptions import JointCases
            if mining_option.split('.')[1] == 'SA':
                result = JointCases.SA(cases)
            else:
                exit(1)
        elif mining_option.split('.')[0] == 'mja':
            from MiningOptions import JointActivities
            if mining_option.split('.')[1] == 'EuclideanDist':
                result = JointActivities.EuclideanDist(cases)
            else:
                exit(1)
        else:
            exit(1)
    except Exception as e:
        print(e)

    g = nx.DiGraph()
    for u, conns in result.items():
        for v, value in conns.items():
            # omit self-loops
            if u != v and abs(value) > 0:
                g.add_edge(u, v, weight=value)

    # plot the node degree distribution (non-weighted)
    N_nodes = g.number_of_nodes()
    normalized_node_degree_list = sorted([d / N_nodes for n, d in
        g.degree_iter()], reverse=True)
    # TODO: plotting using bar graph
    # plot the edge weight distribution
    max_edge_weight = max([wt for u, v, wt in g.edges_iter(data='weight')])
    normalized_edge_weight_list = sorted([wt / max_edge_weight for u, v, wt in
        g.edges.data('weight')], reverse=True)
    # TODO: plotting using line graph

    # write graph
    filetype = '.gml' 
    print('Exporting resulting social network as format' +
            ' (*{}) to file:\t{}'.format(filetype, fout_social_network) + filetype)
    nx.write_gml(g, fout_social_network + filetype)

