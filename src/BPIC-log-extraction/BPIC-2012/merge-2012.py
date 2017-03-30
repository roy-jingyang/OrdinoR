#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

# Working on a converted CSV format data (conversion provided by Disco)
# Data already filtered in Disco.

import sys
import csv
from collections import defaultdict
from datetime import datetime

# AL -> CA -> NO -> VA -> NID
AL = 'W_Afhandelen leads' # Following up on incomplete initial submissions
#BF = 'W_Beoordelen fraude' # Investigating suspect fraud cases
CA = 'W_Completeren aanvraag' # Completing pre-accepted applications
NO = 'W_Nabellen offertes' # Seeking additional information during assessment phase
VA = 'W_Valideren aanvraag' # Follow up after transmitting offers to qualified applicants
NID = 'W_Nabellen incomplete dossiers' #  Assessing the application

# Read in a list of case ids to be filtered
filtered_case_ids = set()
with open(sys.argv[1], 'r') as filter_list_f:
    for line in filter_list_f:
        if line.startswith('#') or line in ['', '\n']:
            pass
        else:
            for x in line.strip().split(','):
                filtered_case_ids.add(int(x))

activities = list()
ate = dict()
# mat[ActivityPrev][ActivityNext][ResPrev][ResNext] = duration of Handover period
mat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
with open(sys.argv[2], 'r') as f:
    is_header_line = True
    ln = 0
    for row in csv.reader(f):
        ln += 1
        if is_header_line:
            is_header_line = False
        else:
            case = int(row[0])
            resource = int(row[2])
            timestamp = datetime.strptime(row[3], '%Y/%m/%d %H:%M:%S.%f')
            variant = int(row[5])
            act = row[-2]
            status = row[-1]
            #print(row)
            if case in filtered_case_ids:
                pass
            else:
                if status == 'START':
                    ate.clear()
                    ate['case'] = case
                    ate['variant'] = variant
                    ate['resource'] = resource
                    ate['activity'] = act
                    ate['TS_start'] = datetime.strftime(timestamp, '%Y/%m/%d %H:%M:%S.%f')
                else:
                    if case == ate['case'] and resource == ate['resource'] and \
                            act == ate['activity']:
                        # select the first COMPLETE TS emerges
                        if 'TS_complete' in ate:
                            pass
                        else:
                            ate['TS_complete'] = datetime.strftime(timestamp, '%Y/%m/%d %H:%M:%S.%f')
                            activities.append((ate['case'], ate['variant'], ate['resource'], ate['activity'], ate['TS_start'], ate['TS_complete']))
                    else:
                        print('Line {}: Error happened on processing Case #{}'.format(ln, case))
                        print('Resource #{} on Activity {} @ {}'.format(resource, act), end='')
                        print(timestamp)
                        exit()


with open(sys.argv[3], 'w') as export:
    writer = csv.writer(export)
    writer.writerow(['Case ID', 'Variant Index', 'Resource ID', 'Activity Name', 'Start', 'Complete'])
    for i in range(len(activities)):
        writer.writerow(activities[i])

