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
        p: float, in range [0, 1]
    Returns:
        modes: iterator
            The execution modes corresponding to the resources:
            - if a splitting is not needed, then a frozenset is returned;
            - if a splitting is needed, then a dict is returned, with the
              subgroups as dict keys and the related execution modes as values.
    '''
    modes = _get_modes_by_prop(group, rl, p)
    
    return modes
    '''
    if len(modes) > 0:
        return modes
    else:
        # post refining required
        sigma = dict()
        all_subgroups = _powerset_excluded(group)
       
        # Algorithm Weighted SetCoverGreedy
        while True:
            uncovered = group.difference(
                    set.union(set(sg) for sg in sigma.keys()))

            if len(uncovered) > 0:
                def cost_effectiveness(s):
                    # calculate the cost (weight)
                    m = _get_modes_by_prop(s, rl, p)
                    # TODO: different cost function definitions
                    # Strategy 1. Maximum Capability
                    cost = 1.0 / len(m) if len(m) >= 1 else float('inf')
                    # Strategy 2. Maximum Size
                    #cost = 1.0 / len(s) if len(s) >= 1 else float('inf')
                    # calculate the cost effectiveness
                    return len(uncovered.intersection(s)) / cost

                best_sg = max(all_subgroups, key=cost_effectiveness)
                sigma[best_sg] = _get_modes_by_prop(best_sg)
            else:
                break

        return sigma
    '''

def _get_modes_by_prop(g, rl, p):
    '''Find the executions modes of a given (sub)group, if a certain percentage
    of member resources have executed these modes.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
        p: float, in range [0, 1]
    Returns:
        modes: frozenset
    '''

    modes = set()
    grouped_by_resource = rl.groupby('resource')
    from collections import defaultdict
    count_mode_originator = defaultdict(set)
    for r in g:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            count_mode_originator[m].add(r)
    
    for m, originators in count_mode_originator.items():
        if len(originators) * 1.0 / len(g) >= p:
            modes.add(m)
    return frozenset(modes)

# Python recipe: this is a helper function
from itertools import chain, combinations
def _powerset_excluded(iterable):
    s = list(iterable)
    return (chain.from_iterable(combinations(s, r) 
        for r in range(1, len(s))))

