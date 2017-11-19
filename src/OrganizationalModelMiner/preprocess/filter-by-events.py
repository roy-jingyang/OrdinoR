#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

f_event_log = sys.argv[1]
fout_dir = sys.argv[2]
#fout_event_log = sys.argv[2]

groups = ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 7',
        'Group 12', 'Group 13', 'Group 14', 'Group 15']

# filter by events
with open(f_event_log, 'r') as f, open(fout_dir + '/ALL.csv', 'w') as fout:
    writer = csv.writer(fout)
    is_header_line = True
    for row in csv.reader(f):
        if is_header_line:
            writer.writerow(row)
            is_header_line = False
        else:
            #case_group = row[8]
            org_group = row[13]
            if org_group == 'EMPTY':
                pass
            else:
                writer.writerow(row)

for g in groups:
    fn = 'LG' + g.split()[-1] + '.csv'
    with open(f_event_log, 'r') as f, open(fout_dir + '/' + fn, 'w') as fout:
        writer = csv.writer(fout)
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                writer.writerow(row)
                is_header_line = False
            else:
                #case_group = row[8]
                org_group = row[13]
                if org_group == 'EMPTY' or org_group == g:
                    pass
                else:
                    writer.writerow(row)

