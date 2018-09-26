# -*- coding: utf-8 -*-

'''
1.
This module contains methods for importing event log data. Currently supporting
    [√] Disco-exported CSV format (https://fluxicon.com/disco/)
    [ ] (TODO) eXtensible Event Stream (XES) (http://xes-standard.org/)
    (Solution: opyenxes)
    [ ] (TODO) MXML (ProM 5)
    (Solution: opyenxes)
'''

'''
Data exchange format for an event log - all methods in this module MUST return
the successfully imported event log as a pandas DataFrames:
    'case_id', 'activity', 'resource', 'timestamp',
 0
 1
 .
 .
 .
 n                                           }

Note: 
    The indices are created using integers increasing from 0, each of which
corresponds to one event.
    The original event log data should at least provide the information of the
activity ids. It is expected that the resource ids and timestamps are presented
as well (not mandatory though).
    Other extensible data attributes can be appended according to different 
purposes of working projects and the event log(s) acquired.

2.
This module contains methods for importing an organizational model.
'''

import sys
import csv

# 1. Import an event log
def _describe_event_log(el):
    '''
    Params:
        el: DataFrame
            The event log in pandas DataFrame form.
    Returns:
    '''

    print('-' * 80)

    print('Number of events:\t\t{}'.format(len(el)))
    print('Number of cases:\t\t{}'.format(len(el.groupby('case_id'))))
    print('Event log attributes:\n\t') # TODO

    print('-' * 80)
    return

def read_disco_csv(fn, mapping=None, header=True, encoding='utf-8'):
    '''
    Params:
        fn: str
            Filename of the event log file being imported.
        mapping: dict, optional
            A python dictionary that denotes the mapping from CSV column
            numbers to event log attributes.
        header: boolean, optional
            True if the event log file contains a header line, False otherwise.
        encoding: str, optional
            Encoding of the event log file being imported.
    Returns:
        el: DataFrame
            The event log in pandas DataFrame form.
    '''

    ld = list()
    with open(fn, 'r', encoding=encoding) as f:
        is_header_line = True
        line_count = 0

        for row in csv.reader(f):
            line_count += 1
            if is_header_line:
                is_header_line = False
                pass
            else:
                # the default mapping is defined as below
                e = {
                    'case_id': row[0],
                    'activity': row[1],
                    'resource': row[2],
                    'timestamp': row[3]
                }
                # add addtional attributes mapping specified
                if mapping:
                    for attr, col_num in mapping.items():
                        if attr not in e:
                            e[attr] = row[col_num]

                ld.append(e)

    from pandas import DataFrame
    el = DataFrame(ld)

    print('"{}" imported successfully. {} lines scanned.'.format(
        fn, line_count))

    _describe_event_log(el)
    return el

def read_xes(fn, encoding='utf-8'):
    pass

def read_mxml(fn, encoding='utf-8'):
    pass

# 2. Import an organizational model
# TODO: [Deprecated]
def read_org_model_csv_old(fn, encoding='utf-8'):
    from collections import defaultdict
    model = defaultdict(lambda: set())
    with open(fn, 'r', encoding=encoding) as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for r in row[2].split(';'):
                    model[row[0]].add(r)

    return model

def read_org_model_csv(fn, encoding='utf-8'):
    # TODO
    return om

