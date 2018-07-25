# -*- coding: utf-8 -*-

'''
This module contains the implementation of the default mining method proposed
by Song & van der Aalst (ref. Song & van der Aalst, DSS 2008).
'''

def mine(c):
    '''
    The default mining method.

    Params:
        c: DataFrame
            The imported event log.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''

    print('Applying Default Mining:')
    # group: task_name => [involved originators]
    og = dict()
    for activity, events in c.groupby('activity'):
        og[activity] = set(events['resource'])
    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

