#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains methods for associating mined organizational groups with
tasks, i.e. entity assignment (ref. Song & van der Aalst, DSS 2008).
'''

from collections import defaultdict

def assign(og, D):
    '''
    This is the default method proposed by Song & van der Aalst, DSS 2008.

    Params:
        D: dict of DataFrames
            The imported event log.
        og: dict of sets
            The mined organizational groups.
    Returns:
        a: dict of sets
            The entity assignment result (group => task(s)).
    '''
    a = defaultdict(lambda: set())
    # for each case
    for case_id, trace in D.items():
        # for each event
        for e in trace.itertuples():
            # check if resource in specific group
            for gid, g in og.items():
                if e.resource in g:
                    a[gid].add(e.activity)

    return a

