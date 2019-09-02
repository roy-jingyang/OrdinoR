#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src')
import os
from os import listdir
from os.path import join

dir_split_logs = sys.argv[1]
fnout = 'trace_clustering.result'

if __name__ == '__main__':
    from IO.reader import read_disco_csv
    n_split_logs = 0
    for fn in listdir(dir_split_logs):
        if fn.endswith('.csv'):
            n_split_logs += 1
            with open(fn, 'r', encoding='utf-8') as f, open(
                join(dir_split_logs, fnout), 'w+') as fout:
                el = read_disco_csv(f)
                for case_id in set(el['case_id']):
                    fout.write('{}\t{}\n'.format(case_id, n_split_logs))

