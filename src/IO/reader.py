# -*- coding: utf-8 -*-

'''
This module contains methods for importing event log data. Currently supporting
    [√] Disco-exported CSV format (https://fluxicon.com/disco/)
    [ ] (TODO) eXtensible Event Stream (XES) (http://xes-standard.org/)
    [ ] (TODO) MXML (ProM 5)
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
as well (not mandotory though).
    Other extensible data attributes can be appended according to different 
purposes of working projects and the event log(s) acquired.
'''

import sys
import csv
import pandas as pd

def _describe_event_log(df):
    '''
    Params:
        df: DataFrame
    Returns:
    '''

    print('-' * 80)

    print('Number of events:\t\t{}'.format(len(df)))
    print('Number of cases:\t\t{}'.format(len(df.groupby('case_id'))))
    print('Event log attributes:\n\t') # TODO

    print('-' * 80)
    return

# 1. Disco-exported CSV format event log file
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
        df: DataFrame
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

    df = pd.DataFrame(ld)

    print('"{}" imported successfully. {} lines scanned.'.format(
        fn, line_count))

    _describe_event_log(df)
    return df

# TODO 2. XES format event log file.
def read_xes(fn, encoding='utf-8'):
    pass

# TODO 3. MXML format event log file.
def read_mxml(fn, encoding='utf-8'):
    pass

