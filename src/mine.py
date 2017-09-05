#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

import sys
import csv
import pandas as pd
from collections import defaultdict
from datetime import datetime

if __name__ == '__main__':
    cases = defaultdict(lambda: list())
    # The following read-in part is for BPIC 2013 Incident Mngt. logs only
    with open(sys.argv[1], 'r') as f:
        is_header_line = True
        ln = 0
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                case = int(row[0])
                variant = int(row[1])
                resource = int(row[2])
                activity = row[3]
                TS_start = datetime.strptime(row[4], '%Y/%m/%d %H:%M:%S.%f')
                TS_end = datetime.strptime(row[5], '%Y/%m/%d %H:%M:%S.%f')
                cases[case].append((case, variant, resource, activity, TS_start, TS_end))

    print('Log file loaded successfully. # of cases read: {}'.format(len(cases.keys())))
    print('Average # of activities within each case: {}'.format(sum(
    len(x) for k, x in cases.items()) / len(cases.keys())))
    # read-in ends

    opt = sys.argv[3]
    try:
        if opt.split('.')[0] == 'Subcontracting':
            from MiningOptions import subcontracting
            if opt.split('.')[1] == 'CCCDCM':
                # TODO
                result = subcontracting.CCCDCM(cases)
            else:
                exit(1)
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

    print('Exporting data as adjacency list as json format to file: {}'.format(sys.argv[2]))
    df = pd.DataFrame(result)
    # output as adjacency list
    with open(sys.argv[2], 'w') as fout:
        fout.write(df.to_json())
        fout.write('\n')

