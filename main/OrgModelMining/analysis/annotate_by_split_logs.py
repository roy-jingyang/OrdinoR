#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from os import listdir
from os.path import join

dir_split_logs = sys.argv[1]
fnout = 'trace_clustering.result'

if __name__ == '__main__':
    from orgminer.IO.reader import read_disco_csv, read_xes
    l_split_logs = sorted(
        #fn for fn in listdir(dir_split_logs) if fn.endswith('.csv'))
        fn for fn in listdir(dir_split_logs) if fn.endswith('.xes'))
    print('{} log files in CSV format detected:'.format(len(l_split_logs)))
    print(l_split_logs)
    print()

    results = list()
    for i, fn in enumerate(l_split_logs):
        with open(join(dir_split_logs, fn), 'r', encoding='utf-8') as f:
            #el = read_disco_csv(f)
            el = read_xes(f)
            for case_id in sorted(set(el['case_id'])):
                results.append('{}\t{}\n'.format(case_id, fn.split('.')[0]))

    with open(join(dir_split_logs, fnout), 'w+') as fout:
        for line in results:
            fout.write(line)

