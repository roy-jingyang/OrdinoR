#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from orgminer.IO.reader import read_disco_csv
from orgminer.IO.reader import read_xes

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    with open(filename_input, 'r') as f:
        #el = read_disco_csv(f)
        #el = read_disco_csv(f, header=False)
        el = read_xes(f)

    print(el)
    print(type(el))
    print(len(el))

    print(el.index)
    print(el.columns)

    '''
    # iterate by each case and then by each event within the current case (as
    # Series)
    print(el.groupby('case_id').groups.keys())
    print(len(el.groupby('case_id')))
    for case_id, events in el.groupby('case_id'):
        performers = set(events['resource'])
        print(performers)
        exit()
    #print(el.groupby('resource').groups.keys())
    
    om = read_org_model_csv(filename_input)
    print(om)
    print(om._rg)
    '''

