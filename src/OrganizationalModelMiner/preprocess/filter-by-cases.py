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
    case_to_filter = set()
    for row in csv.reader(f):
        if is_header_line:
            is_header_line = False
        else:
            case_id = row[0]
            '''
            case_group = row[8]
            if case_group == '':
                case_to_filter.add(case_id)
            else:
                pass
            '''
            org_group = row[13]
            if org_group == 'EMPTY':
                case_to_filter.add(case_id)
            else:
                pass

    print(len(case_to_filter))

    f.seek(0)
    is_header_line = True
    for row in csv.reader(f):
        if is_header_line:
            writer.writerow(row)
            is_header_line = False
        else:
            case_id = row[0]
            if case_id in case_to_filter:
                pass
            else:
                writer.writerow(row)

