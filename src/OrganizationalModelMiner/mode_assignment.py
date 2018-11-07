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
        print('\t[Warning] Group number may change due to refinement.')
        # TODO: different cost function definitions
        def cost(g):
            m = _get_modes_by_prop(frozenset(g), inv_resource_cap, p)
            if len(m) >= 1:
                #return 1 # Naive
                return 1 / len(g) # Max. Size
                #return 1 / len(m) # Max. Cap.
            else:
                return float('inf')

        split = _set_cover_greedy(group, cost, pruning=True)

        print('\tGroup of size {} split to {} subgroups after refinement.'
            .format(len(group), len(split)))
        sigma = dict()
        for subgroup in split:
            subgroup_modes = _get_modes_by_prop(subgroup, inv_resource_cap, p)
            sigma[subgroup] = subgroup_modes
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

def _set_cover_greedy(U, f_cost, pruning=False):
    sigma = list()

    while True:
        if len(sigma) > 0:
            uncovered = U.difference(frozenset.union(*sigma))
        else:
            uncovered = U
        #print('\t\t{}/{} uncovered.'.format(len(uncovered), len(U)))

        if len(uncovered) > 0:
            def cost_effectiveness(g):
                cost = f_cost(g)
                if cost != float('inf'):
                    # valid candidate
                    return len(uncovered.intersection(frozenset(g))) / cost
                else:
                    # invalid candidate
                    return 0

            if pruning:
                candidates = _powerset_exclude_headtail(
                        U, reverse=True, depth=1)
                def find_best_candidate(S):
                    # TODO: debugging required, cannot reproduce
                    best = None
                    best_coverable = None
                    best_cost_effectiveness = float('-inf')
                    best_coverablity = 0
                    for cand in S:
                        num_covered = len(uncovered.intersection(frozenset(cand)))
                        if num_covered > best_coverablity:
                            print('{}-{}'.format(num_covered,
                                best_coverablity))
                            best_coverablity = num_covered
                            best_coverable = cand

                        c_e = cost_effectiveness(cand)
                        if c_e > 0 and c_e > best_cost_effectiveness:
                            best_cost_effectiveness = c_e
                            best = cand

                    if best is not None:
                        return best
                    else:
                        print('recursion takes place')
                        exit()
                        return find_best_candidate(_powerset_exclude_headtail(
                            best_coverable, reverse=True, depth=1))

                best = find_best_candidate(candidates)
            else:
                candidates = _powerset_exclude_headtail(
                        U, reverse=True)
                best = max(candidates, key=cost_effectiveness)

            sigma.append(frozenset(best))
        else:
            return sigma


# Python recipe: this is a helper function
# This function returns a generator Powerset(s) \ {emptyset, s} given a set s

# NOTE 1: the generator delivers subsets based on cardinality on an ascending
# order, specify the additional argument 'reverse' to change the behaviour.
# NOTE 2: the additional argument 'depth' specifies the maximal (minimal) 
# cardinality of subset(s) returned by the function. If None, all will be
# returned.
from itertools import chain, combinations
def _powerset_exclude_headtail(iterable, reverse=False, depth=None):
    s = list(iterable)
    if reverse:
        end = 0 if depth is None else (len(s) - 1 - depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(len(s) - 1, end, -1)))
    else:
        end = len(s) if depth is None else (1 + depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(1, end)))

