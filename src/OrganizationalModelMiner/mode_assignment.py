# -*- coding: utf-8 -*-

'''
This module contains methods for associating mined organizational groups with
tasks.
'''

from collections import defaultdict

def entity_assignment(og, c):
    '''
    This is the default method proposed by Song & van der Aalst, DSS 2008.

    Params:
        og: dict of sets
            The mined organizational groups.
        c: DataFrame
            The imported event log.
    Returns:
        a: dict of sets
            The entity assignment result (group => task(s)).
    '''

    a = dict()
    grouped_by_resource = c.groupby('resource')

    # for each organizational group
    for gid, g in og.items():
        # for each resource in the group
        associated_tasks = set()
        for r in g:
            associated_tasks.update(
                    set(grouped_by_resource.get_group(r)['activity']))
        a[gid] = associated_tasks

    return a

