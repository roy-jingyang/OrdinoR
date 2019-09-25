#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
from datetime import datetime

f_event_log = sys.argv[1]
f_threshold = sys.argv[2]
dir_event_stream_files = sys.argv[3]

if __name__ == '__main__':
    thresholds = list()
    with open(f_threshold, 'r') as ft:
        for line in ft:
            thresholds.append(datetime.strptime(line.strip(), '%Y/%m/%d'))

    el_parts = list()
    for i in range(len(thresholds) - 1):
        el_parts.append(list())

    with open(f_event_log, 'r', encoding='windows-1252') as f:
        is_header_line = True
        ln = 0
        header_line = ''
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                header_line = row
                is_header_line = False
            else:
                caseid = row[0]
                ctimestamp = row[3]
                resource = row[2]
                activity = row[1]
                ts = datetime.strptime(ctimestamp, '%Y/%m/%d %H:%M:%S.%f')
                for i in range(len(thresholds) - 1):
                    if ts > thresholds[i] and ts <= thresholds[i + 1]:
                        el_parts[i].append(row)
                        break

    for i in range(len(thresholds) - 1):
        fn = '{}_part{}_'.format(f_event_log.split('/')[-1].split('.')[0], i) + \
                thresholds[i].strftime('%Y%m%d') + '-' + \
                thresholds[i + 1].strftime('%Y%m%d') + '.csv'
        with open(dir_event_stream_files + '/' + fn, 'w+') as w:
            csv.writer(w).writerow(header_line)
            csv.writer(w).writerows(el_parts[i])

