# -*- coding: utf-8 -*-

'''
This module contains methods for associating discovered organizational groups
with execution modes.
'''

import cProfile

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
    
    print('Applying "assign by any" for mode assignment:')
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
            - if a splitting is needed, then a dict of frozensets is returned,
              with the subgroups as dict keys and the related execution modes
              as values.
    '''
    print('Applying "assign by all" for mode assignment:')
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
            - if a splitting is needed, then a dict of frozensets is returned,
              with the subgroups as dict keys and the related execution modes
              as values.
    '''
    print('Applying "assign by proportion {:.0%}" for mode assignment:'.format(
        p))

    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events
    # pre-computing {mode -> candidate resources} (inv. of Resource Capibility)
    inv_resource_cap = dict()
    for m, events in rl.groupby(['case_type', 'activity_type', 'time_type']):
        inv_resource_cap[m] = set(events['resource'])

    modes = _get_modes_by_prop(group, inv_resource_cap, p)
    
    if len(modes) > 0:
        return modes
    else:
        # post refining required
        # TODO: need to SPEED UP!!
        print('\t[Warning] Group number may change due to refinement.')
        sigma = dict()

        # Algorithm Weighted SetCoverGreedy
        while True:
            if len(sigma) > 0:
                uncovered = group.difference(frozenset.union(*sigma.keys()))
            else:
                uncovered = group

            print('\t\t{}/{} uncovered.'.format(len(uncovered), len(group)))
            if len(uncovered) > 0:
                # NOTE: different cost function definitions
                def cost_effectiveness(g):
                    m = _get_modes_by_prop(
                            frozenset(g), inv_resource_cap, p)
                    if len(m) >= 1:
                        # valid candidate
                        #cost = 1.0 # Strategy Naive
                        #cost = 1.0 / len(g) # Strategy Max. Size
                        cost = 1.0 / len(m) # Strategy Max. Cap
                        return len(uncovered.intersection(frozenset(g))) / cost
                    else:
                        # invalid candidate
                        return float('-inf')

                best = max(_powerset_excluded(group), key=cost_effectiveness)
                '''
                # Pruning for Optimization
                # TODO: Pruning (by Cardinality) only applies to Naive and Max. Size
                prev_cardinality = -1
                best = None
                best_cost_effectiveness = -1
                for cand in _powerset_excluded(group):
                    if prev_cardinality == -1:
                        prev_cardinality = len(cand)

                    if len(cand) < prev_cardinality:
                        if best is None:
                            prev_cardinality = len(cand)
                        else:
                            break

                    c_e = cost_effectiveness(cand)
                    if c_e > 0 and c_e > best_cost_effectiveness:
                        best_cost_effectiveness = c_e
                        best = cand
                '''

                sigma[frozenset(best)] = _get_modes_by_prop(
                        frozenset(best), inv_resource_cap, p)
            else:
                break

        print('\tGroup of size {} split to {} subgroups after refinement.'
            .format(len(group), len(sigma)))
        return sigma

def _get_modes_by_prop(g, inverse_resource_capability, p):
    '''Find the executions modes of a given (sub)group, if a certain percentage
    of member resources have executed these modes.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
        p: float, in range (0, 1]
    Returns:
        modes: frozenset
    '''

    group_size = len(g)
    modes = set()

    for m, resources in inverse_resource_capability.items():
        if len(g.intersection(frozenset(resources))) / group_size >= p:
            modes.add(m)

    return frozenset(modes)

# Python recipe: this is a helper function
# This function returns a generator Powerset(s) \ {emptyset, s} given a set s
# NOTE: the generator starts from delivering the larger subsets, and ends up
# with the single-element subsets (i.e. descending order based on cardinality)
from itertools import chain, combinations
def _powerset_excluded(iterable):
    s = list(iterable)
    return (chain.from_iterable(combinations(s, r) 
        for r in range(len(s) - 1, 0, -1)))

