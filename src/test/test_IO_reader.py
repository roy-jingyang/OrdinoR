#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv

# List input parameters from shell
filename_event_log = sys.argv[1]

if __name__ == '__main__':
    d = read_disco_csv(filename_event_log)
    print(type(d))
    print(sum(len(df) for k, df in d.items()))

