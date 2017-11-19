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
additional_params = sys.argv[4:] if len(sys.argv) > 4 else None

if __name__ == '__main__':
    cases = defaultdict(lambda: list())
    with open(f_event_log, 'r', encoding='windows-1252') as f:
        is_header_line = False
        ln = 0
        # BPiC 2011
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0]
                ctimestamp = row[2]
                resource = row[-1] # only 'org:group' provided
                activity = row[1]
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        '''
        # BPiC 2012
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0]
                ctimestamp = row[3]
                resource = 'EMPTY' if row[2] == '' else row[2]
                activity = row[1]
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        '''
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
        '''

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))

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
        elif mining_option.split('.')[1] == 'CorrelationCoefficient':
            threshold_value = float(additional_params[0])
            result = JointActivities.CorrelationCoefficient(cases, threshold_value)
        else:
            exit(1)
    else:
        exit(1)

    # TODO: directed/undirected, depending on the types of relationships
    #g = nx.DiGraph()
    g = nx.Graph()
    for u, conns in result.items():
        for v, value in conns.items():
            # omit self-loops
            if u != v:
                g.add_node(u)
                g.add_node(v)
                if value is not None:
                    g.add_edge(u, v, weight=value)

    print('#Nodes = {}'.format(len(g)))
    print('{} ({:.2%}) edges'.format(
        len(g.edges), len(g.edges) / (len(g) * (len(g) - 1))))
    import numpy as np
    print('<k> = {:.2f}'.format(np.mean([x[1] for x in g.degree()])))
    # TODO: plot the node degree distribution (non-weighted)
    # TODO: plotting using bar graph
    # plot the edge weight distribution
    # TODO: plotting using line graph

    # write graph
    filetype = '.gml' 
    print('Exporting resulting social network as format' +
            ' (*{}) to file:\t{}'.format(filetype, fout_social_network) + filetype)
    nx.write_gml(g, fout_social_network + filetype)

