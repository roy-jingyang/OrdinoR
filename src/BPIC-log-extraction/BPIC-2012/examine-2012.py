#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

# Working on a converted CSV format data (conversion provided by Disco)
# Data already filtered.

import sys
import csv
from collections import defaultdict, deque, Counter
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

case_id_set = set()
cases = defaultdict(lambda: list())
merged_events = list()
activity = dict()

with open(sys.argv[1], 'r') as f:
    is_header_line = True
    for row in csv.reader(f):
        if is_header_line:
            is_header_line = False
        else:
            case = int(row[0])
            resource = int(row[2]) if row[2] != '' else ''
            timestamp = datetime.strptime(row[3], '%Y/%m/%d %H:%M:%S.%f')
            variant = int(row[5]) if row[5] != '' else ''
            act = row[-2] if row[-2] != '' else ''
            status = row[-1] if row[-1] != '' else ''
            #print(row)
            cases[case].append((resource, act, status, timestamp))
            '''
            if case not in case_id_set:
                activity.clear()
            else:
                if status == 'START':
                    merged_events.append([case, variant, act, resource, \
                    activity['start'], activity['complete']])
                    activity.clear()
                    activity['start'] = timestamp
                    activity['name'] = act
                    activity['resource'] = resource
                else:
                    activity['complete'] = timestamp
                    if act != activity['name'] or resource != activity['resource']:
                        print(case)
                        exit()
            '''

# Examine data
for case, trace in cases.items():
    #activity = deque(list(), maxlen=1)
    #start_activity = list()
    activity = Counter()
    not_started_activity = list()
    for i in range(len(trace)):
        if trace[i][2] == 'START':
            '''
            # 3 checking not closed activity
            start_activity.append((trace[i][0], trace[i][1], trace[i][-1], i))
            '''
            activity[(trace[i][0], trace[i][1])] += 1
            '''
            # 2 checking overlapping
            if len(activity) == 0:
                activity.append(trace[i])
            else:
                print('Possible overlapping for Case #{}'.format(case) + \
                ' @ Event #{} {}'.format(i+1, trace[i][1]))
                print('Current flag:\t', end='')
                print(activity[0])
                print('Current found:\t', end='')
                print(trace[i])
                print('\n')
                activity.pop()
            '''
        else:
            '''
            # 3 checking not closed activity
            start_act_index = None
            for j in range(len(start_activity)):
                if trace[i][0] == start_activity[j][0] and trace[i][1] == start_activity[j][1]:
                    start_act_index = j
                    break
            if start_act_index is not None:
                start_activity.pop(start_act_index)
            '''
            '''
            # 2 checking overlapping
            if trace[i][0] == activity[0][0] and trace[i][1] == activity[0][1]:
                activity.pop()
        else:
            pass
            '''
            if activity[(trace[i][0], trace[i][1])] < 1:
                not_started_activity.append((trace[i][0], trace[i][1], i))

    '''
    for not_closed_act in start_activity:
        print('Possible not closed activity for Case #{} @ Event #{} {}'.format(case, not_closed_act[-1] + 1, not_closed_act[1]))
    '''
    '''
    for not_started_act in not_started_activity:
        print('Possible missing start trans for Case # {} @ Event #{} {}'.format(case, not_started_act[-1] + 1, not_started_act[1]))
    '''

    '''
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
    '''
