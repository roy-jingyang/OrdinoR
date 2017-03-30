#! /usr/bin/env python3
#! -*- coding: utf-8 -*-

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

cases = defaultdict(lambda: list())
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

