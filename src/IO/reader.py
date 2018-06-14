#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains methods for importing event log data. Currently supporting
    [√] Disco-exported CSV format (https://fluxicon.com/disco/)
    [ ] (TODO) eXtensible Event Stream (XES) (http://xes-standard.org/)
    [ ] (TODO) MXML (ProM 5)
'''

'''
Data exchange format for an event log - all methods in this module MUST return
the successfully imported event log as a Python dict of pandas DataFrames:
    [case_id]: {  'activity', 'resource', 'timestamp',
                 0
                 1
                 .
                 .
                 .
                 n                                           }
    [case_id]: ...

Note: 
    The original event log data should at least provide the information of the
activity ids. It is expected that the resource ids and timestamps are presented
as well (not mandotory though).
    Other extensible data attributes can be appended according to different 
purposes of working projects and the event log(s) acquired.
'''

import sys
import csv
import pandas as pd
from collections import defaultdict

def _describe_event_log(D):
    '''
    Params:
        D: dict of DataFrames
    Returns:

    '''
    print('-' * 80)

    print('Number of cases:\t\t{}'.format(len(D)))

    print('-' * 80)
    return

# 1. Disco-exported CSV format event log file
def read_disco_csv(fn, header=True, encoding='utf-8'):
    '''
    Params:
        fn: str
            Filename of the event log file being imported.
        header: Boolean, optional
            True if the event log file contains a header line, False otherwise.
        encoding: str, optional
            Encoding of the event log file being imported.
    Returns:
        D: dict of DataFrames 
    '''
    fields = [
            'activity',
            'resource',
            'timestamp'
            ]

    D = defaultdict(lambda: pd.DataFrame(columns=fields))

    with open(fn, 'r', encoding=encoding) as f:
        is_header_line = True
        line_count = 0

        for row in csv.reader(f):
            line_count += 1
            if is_header_line:
                is_header_line = False
                pass
            else:
                case_id = row[0]
                e = pd.Series(row[1:4])

                D[case_id] = D[case_id].append(e, ignore_index=True)

    print('"{}" imported successfully. {} lines scanned.'.format(fn, line_count))

    _describe_event_log(D)
    return D

# TODO 2. XES format event log file.
def read_xes(fn, encoding='utf-8'):
    pass

# TODO 3. MXML format event log file.
def read_mxml(fn, encoding='utf-8'):
    pass

