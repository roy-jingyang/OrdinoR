#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv

# List input parameters from shell
filename_event_log = sys.argv[1]

if __name__ == '__main__':
    c = read_disco_csv(filename_event_log)
    #print(c)
    print(type(c))
    # TODO
    # iterate by each case and then by each event within the current case (as
    # Series)
    print(len(c.groupby('case_id')))
    #for case_id, events in c.groupby('case_id'):
    #    performers = set(events['resource'])
    #    print(performers)
    #    exit()
    print(c.groupby('resource').groups.keys())

