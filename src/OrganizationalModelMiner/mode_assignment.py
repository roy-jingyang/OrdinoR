# -*- coding: utf-8 -*-

'''
This module contains methods for associating mined organizational groups with
tasks.
'''

from collections import defaultdict

def default_assign(group, rl):
    '''
    This is the default method proposed by Song & van der Aalst, DSS 2008,
    namely "entity_assignment".

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
    Returns:
        modes: iterator
            The execution modes corresponding to the resources.
    '''

    modes = set()
    grouped_by_resource = rl.groupby('resource')

    # for each resource in the group
    for r in group:
        for event in grouped_by_resource.get_group(r)[
                ['case_type', 'activity_type', 'time_type']].itertuples():
            modes.add((event.case_type, event.activity_type, event.time_type))
    return modes

