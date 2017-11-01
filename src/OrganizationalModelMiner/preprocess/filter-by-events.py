#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

f_event_log = sys.argv[1]
fout_event_log = sys.argv[2]

# filter by events
with open(f_event_log, 'r') as f, open(fout_event_log, 'w') as fout:
    writer = csv.writer(fout)
    is_header_line = True
    for row in csv.reader(f):
        if is_header_line:
            writer.writerow(row)
            is_header_line = False
        else:
            '''
            case_group = row[8]
            if case_group == '':
                pass
            else:
                writer.writerow(row)
            '''

            org_group = row[13]
            if org_group == 'EMPTY':
                pass
            else:
                writer.writerow(row)

