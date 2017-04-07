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

# Mining options
# Possible Metrics to be chosen
SUBCONTRACTING = 0
HANDOVER_OF_WORK = 1000
WORKING_TOGETHER = 2000
SIMILAR_TASK = 3000
#REASSIGNMENT = 4000
#SETTINGS = 5000
# Constants for 'subcontracting/handover' of work setting
CONSIDER_CAUSALITY = 100
CONSIDER_DIRECT_SUCCESSION = 10
CONSIDER_MULTIPLE_TRANSFERS = 1
# Constants for 'working together' setting
SIMULTANEOUS_APPERANCE_RATIO = 0
DISTANCE_WITHOUT_CAUSALITY = 1
DISTANCE_WITH_CAUSALITY = 2
# Constants for 'similar task' setting
EUCLIDEAN_DISTANCE = 0
CORRELATION_COEFFICIENT = 1
SIMILARITY_COEFFICIENT = 2
HAMMING_DISTANCE = 3
# Constants for 'reassignment' setting
'''
MULTIPLE_REASSIGNMENT = 1
'''
# Constants for 'settings_grouping' setting
'''
GROUP_BY_ORG_UNIT = 0
GROUP_BY_ROLE = 1
GROUP_BY_ORG_UNIT_ROLE = 2
'''

if __name__ == '__main__':
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

    opt = sys.argv[2]
    try:
        if opt.split('.')[0] == 'Subcontracting':
            from MiningOptions import subcontracting
            if opt.split('.')[1] == 'CCCDCM':
                # TODO
                result = subcontracting.Subcontracting_CCCDCM(cases)
            else:
                exit(1)
        elif opt.split('.')[0] == 'Handover':
            from MiningOptions import handover
            if opt.split('.')[1] == 'CCCDCM':
                result = handover.CCCDCM(cases)
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

    print('# of emerged resources: {}'.format(len(result)))

