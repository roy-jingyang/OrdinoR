#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

# Working on a converted CSV format data (conversion provided by Disco)
# Data already filtered.

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

# mat[ActivityPrev][ActivityNext][ResPrev][ResNext] = duration of Handover period
mat = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

event_start = None
with open(sys.argv[1], 'r') as f:
    is_header_line = True
    for row in csv.reader(f):
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
            if status == 'START':
                event_start = (resource, act, timestamp)
            else:
                if event_start is not None:
                    dur = timestamp - event_start[2]
                    #print(dur)
                    mat[event_start[1]][act][event_start[0]][resource] = dur
                    event_start = None

'''
count = 0
for case, trace in cases.items():
    #acts = [AL, CA, NO, VA, NID]
    index = list()
    for i in range(len(trace)):
        if trace[i] in acts:
            index.append(acts.index(trace[i]))
    if all(index[i] <= index[i+1] for i in range(len(index) - 1)):
        pass
    else:
        if trace.index(NO) > trace.index(VA):
            print(case)
            count += 1
print(count)
'''

