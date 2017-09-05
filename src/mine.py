#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import sys
import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx

if __name__ == '__main__':
    cases = defaultdict(lambda: list())
    # The following read-in part is for BPIC 2013 Incident Mngt. logs only
    # Log format:
    # SR Number;Change Date+Time;Status;Sub Status;Involved ST Function Div;
    # Involved Org line 3;Involved ST;SR Latest Impact;
    # Product;Country;Owner Country;Owner First Name
    with open(sys.argv[1], 'r', encoding='windows-1252') as f:
        is_header_line = True
        ln = 0
        for line in f:
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                row = line.split(';')
                case = row[0] # SR Number
                cdatetime = row[1] # Change Date+Time
                resource = row[-1]
                activity = row[2] + row[3]
                cases[case].append((case, cdatetime, resource, activity))

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))
    # read-in ends

    opt = sys.argv[3]
    try:
        if opt.split('.')[0] == 'Subcontracting':
            # TODO
            print('Under construction: [TODO]')
            exit(1)
            '''
            from MiningOptions import subcontracting
            if opt.split('.')[1] == 'CCCDCM':
                result = subcontracting.CCCDCM(cases)
            else:
                exit(1)
            '''
        elif opt.split('.')[0] == 'Handover':
            from MiningOptions import handover
            if opt.split('.')[1] == 'ICCDCM':
                # Our main concern now
                # handover.ICCDCM(cases=list_of_cases, is_task_specific=false)
                result = handover.ICCDCM(cases)
            elif opt.split('.')[1] == 'ICCDIM':
                # inapplicable for time based calculation
                result = handover.ICCDIM(cases)
            elif opt.split('.')[1] == 'ICIDCM':
                # inapplicable for time based calculation
                result = handover.ICIDCM(cases)
            elif opt.split('.')[1] == 'ICIDIM':
                # inapplicable for time based calculation
                result = handover.ICIDIM(cases)
            else:
                exit(1)
        elif opt.split('.')[0] == 'WorkingTogether':
            from MiningOptions import workingtogether
            if opt.split('.')[1] == 'SAR':
                result = workingtogether.SAR(cases)
            else:
                exit(1)
        elif opt.split('.')[0] == 'SimilarTask':
            from MiningOptions import similartask
            if opt.split('.')[1] == 'ED':
                result = similartask.ED(cases)
            else:
                exit(1)
        else:
            exit(1)
    except Exception as e:
        print(e)

    print('Exporting resulting social network as GML format (*.gml) to file: {}'.format(sys.argv[2]))
    g = nx.DiGraph()
    for u, conns in result.items():
        for v, value in conns.items():
            if value > 0:
                g.add_edge(u, v, weight=value)
    nx.write_gml(g, sys.argv[2])

