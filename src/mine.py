#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import sys
import csv
from collections import defaultdict
from datetime import datetime
import networkx as nx

if __name__ == '__main__':
    cases = defaultdict(lambda: list())
    with open(sys.argv[1], 'r', encoding='windows-1252') as f:
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

    filetype = '.gml' 
    print('Exporting resulting social network as format' +
            ' (*{}) to file:\t{}'.format(filetype, sys.argv[2]) + filetype)
    g = nx.DiGraph()
    for u, conns in result.items():
        for v, value in conns.items():
            g.add_edge(u, v, weight=value)
    nx.write_gml(g, sys.argv[2] + filetype)

