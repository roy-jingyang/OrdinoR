#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

# import methods to be tested below
from IO.reader import disco_csv_read

# List input parameters from shell
filename_event_log = sys.argv[1]

if __name__ == '__main__':
    d = read_disco_csv(filename_event_log)
    print(type(d))

