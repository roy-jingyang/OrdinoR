# -*- coding: utf-8 -*-

'''
This module contains methods for associating discovered organizational groups
with execution modes.
'''

def member_first_assign(group, rl):
    '''Assign an execution mode to a group, as long as there exists a member
    resource of this group that have executed this mode, i.e. everything done
    by each member matters.

    Note: this is the method proposed by Song & van der Aalst, DSS 2008,
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

    for r in group:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            modes.add(m)
    return frozenset(modes)

def group_first_assign(group, rl):
    '''Assign an execution mode to a group, only if every member resources of
    this group have executed this mode, i.e. only things done by every member
    matter.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
    Returns:
        modes: iterator
            The execution modes corresponding to the resources.
    '''
    return vote_assign(group, rl, percentage=1.0)

def vote_assign(group, rl, percentage):
    '''Assign an execution mode to a group, only if certain percentage of
    member resources of this group have executed this mode, i.e.only things
    done by certain percentage of members matter.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
        percentage: float, in range (0, 1]
    Returns:
        modes: iterator
            The execution modes corresponding to the resources.
    '''
    modes = set()
    grouped_by_resource = rl.groupby('resource')
    from collections import defaultdict
    count_mode_originator = defaultdict(lambda: set())
    for r in group:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            count_mode_originator[m].add(r)
    
    for m, originators in count_mode_originator.items():
        if len(originators) * 1.0 / len(group) >= percentage:
            modes.add(m)

    return frozenset(modes)

