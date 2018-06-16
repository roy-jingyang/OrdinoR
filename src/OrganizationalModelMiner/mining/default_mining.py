#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of the 'default mining' method proposed
by Song & van der Aalst (ref. Song & van der Aalst, DSS 2008).
'''

import copy
from numpy import unique

def mine(c):
    '''
    The 'default mining' method.
    Params:
        c: DataFrame
            The imported event log.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''

    print('Applying Default Mining:')
    print('{} cases to be processed.'.format(len(c.groupby('case_id'))))
    # group: task_name => [involved originators]
    og = dict()
    for activity, events in c.groupby('activity'):
        og[activity] = set(events['resource'])
    print('{} organizational entities extracted.'.format(len(og)))
    return copy.deepcopy(og)

