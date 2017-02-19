#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import time
from collections import defaultdict

# Merging

# Incident (28 cols)

# CI Name (aff); CI Type (aff); CI Subtype (aff); Service Component WBS (aff); 
# Incident ID; Status; Impact; Urgency; 
# Priority; Category; KM number; Alert Status; 
# # Reassignments; Open Time; Reopen Time; Resolved Time; 
# Close Time; Handle Time (Hours); Closure Code; # Related Interactions; 
# Related Interaction; # Related Incidents; # Related Changes; Related Change; 
# CI Name (CBy);  CI Type (CBy); CI Subtype (CBy); ServiceComp WBS (CBy)

Incidents = defaultdict(lambda: dict())
n_cols1 = 0
with open(sys.argv[1], 'r') as f1:
    pass_header = False
    for line in f1:
        if pass_header:
            row = line.strip().split(';')
            if not row[4] == '':
                Incidents[row[4]]['Status'] = row[5]
                Incidents[row[4]]['Impact'] = row[6]
                Incidents[row[4]]['Urgency'] = row[7]
                Incidents[row[4]]['Priority'] = row[8]
                Incidents[row[4]]['Category'] = row[9]
                Incidents[row[4]]['AlertStatus'] = row[11]
                Incidents[row[4]]['Closure'] = row[18]
                for k, v in Incidents[row[4]].items():
                    Incidents[row[4]][k] = None if '' == v else v

                Incidents[row[4]]['NReassignments'] = row[12]
                Incidents[row[4]]['NRelatedInteractions'] = row[19]
                Incidents[row[4]]['NRelatedIncidents'] = row[21]
                Incidents[row[4]]['NRelatedChanges'] = row[22]
                for k, v in Incidents[row[4]].items():
                    Incidents[row[4]][k] = 0 if '' == v else v

                Incidents[row[4]]['Open'] = None if '' == row[13] \
                        else time.strptime(row[13], '%d/%m/%Y %H:%M:%S')
                Incidents[row[4]]['Reopen'] = None if '' == row[14] \
                        else time.strptime(row[14], '%d/%m/%Y %H:%M:%S')
                Incidents[row[4]]['Resolved'] = None if '' == row[15] \
                        else time.strptime(row[15], '%d/%m/%Y %H:%M:%S')
                Incidents[row[4]]['Close'] = None if '' == row[16] \
                        else time.strptime(row[16], '%d/%m/%Y %H:%M:%S')
        else:
            pass_header = True
            header = line.strip().split(';')
            for i in header:
                n_cols1 += 0 if '' == i else 1
print('{} lines read from {}'.format(len(Incidents), sys.argv[1]))

# Incident-Activity (8 cols)

# Incident ID;IncidentActivity_Type;Assignment Group;Timestamp
# Variant;IncidentActivity_Number;KM number;Interaction ID

n_cols2 = 0
missing_incidents = 0
with open(sys.argv[2], 'r') as f2:
    pass_header = False
    for line in f2:
        if pass_header:
            row = line.strip().split(';')
            if row[0] not in Incidents:
                missing_incidents += 1
            else:
                if 'Activities' not in Incidents[row[0]]:
                    Incidents[row[0]]['Activities'] = list()
                Incidents[row[0]]['Activities'].append((row[5], row[1]))
                Incidents[row[0]]['Group'] = row[2]
                Incidents[row[0]]['Variant'] = row[4]
        else:
            pass_header = True
            header = line.strip().split(';')
            for i in header:
                n_cols2 += 0 if '' == i else 1

# Encoding & Preprocessing


# Testing

keys = list(Incidents.keys())
#for k, v in Incidents[keys[0]].items():
for k, v in Incidents.items():
    if v['Close'] is None and v['Reopen'] is not None and v['Reopen'] > v['Close']:
        print(v)
    '''
    print(k, end='\t')
    print(v)
    '''

