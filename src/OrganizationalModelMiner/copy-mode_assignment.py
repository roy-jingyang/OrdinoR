# -*- coding: utf-8 -*-

'''
This module contains methods for associating discovered organizational groups
with execution modes.
'''

def assign_by_any(group, rl):
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

def assign_by_all(group, rl):
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
            The execution modes corresponding to the resources:
            - if a splitting is not needed, then a frozenset is returned;
            - if a splitting is needed, then a dict is returned, with the
              subgroups as dict keys and the related execution modes as values.
    '''
    return assign_by_proportion(group, rl, p=1.0)

def assign_by_proportion(group, rl, p):
    '''Assign an execution mode to a group, only if certain percentage of
    member resources of this group have executed this mode, i.e.only things
    done by certain percentage of members matter.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
        p: float, in range (0, 1]
    Returns:
        modes: iterator
            The execution modes corresponding to the resources:
            - if a splitting is not needed, then a frozenset is returned;
            - if a splitting is needed, then a dict is returned, with the
              subgroups as dict keys and the related execution modes as values.
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
        if len(originators) * 1.0 / len(group) >= p:
            modes.add(m)

    if len(modes) > 0:
        return frozenset(modes)
    else:
        # TODO
        # post refining required
        # weighted SetCoverGreedy
        sigma = dict()
        uncovered = group.difference(set(sigma.keys()))
        all_subgroups = _powerset_excluded(group)
        def cost_effectiveness(s):
            # calculate the cost (weight)

            # calculate the cost effectiveness
            return len(uncovered.intersection(s)) / cost

        while set(sigma.keys()) != group:
            best_s = None
            best_cost_effectiveness = float('-inf')
            # find the one with best cost effectiveness
            all_cost_effectiveness = map(cost_effectiveness, all_subgroups)

        return sigma

# Python recipe: this is a helper function
from itertools import chain, combinations
def _powerset_excluded(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) 
        for r in range(1, len(s)))

