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
    modes = _get_modes_by_prop(group, rl, p)
    
    if len(modes) > 0:
        return modes
    else:
        # post refining required
        print('\t[Warning] Applying refinement, may increase group number.')
        print('\tGroup of size {} is being refined: '.format(
            len(group)), end='')
        def cost(s):
            m = _get_modes_by_prop(s, rl, p)
            # TODO: different cost function definitions
            # Strategy 1. Maximum Capability
            cost = 1.0 / len(m) if len(m) >= 1 else None
            # Strategy 2. Maximum Size
            #cost = 1.0 / len(s) if len(m) >= 1 else None
            return cost, m

        sigma = dict()

        # pre-calculate the cost and execution modes for all subgroups

        from collections import defaultdict
        all_candidate_subgroups = defaultdict(dict)
        for subgroup in _powerset_excluded(group):
            group_cost, modes = cost(frozenset(subgroup))
            if group_cost is not None:
                k = tuple(sorted(subgroup))
                all_candidate_subgroups[k]['cost'] = group_cost
                all_candidate_subgroups[k]['modes'] = modes
            else:
                pass

        print('\t\t{} candidate subgroups under test'.format(
            len(all_candidate_subgroups)))
        # Algorithm Weighted SetCoverGreedy
        while True:
            if len(sigma) > 0:
                uncovered = group.difference(frozenset.union(*sigma.keys()))
            else:
                uncovered = group

            print('\t\t{}/{} uncovered'.format(len(uncovered), len(group)))

            if len(uncovered) > 0:
                def cost_effectiveness(k):
                    return (len(uncovered.intersection(frozenset(k))) /
                            all_candidate_subgroups[k]['cost'])

                best_sg = max(
                        all_candidate_subgroups.keys(), key=cost_effectiveness)
                sigma[frozenset(best_sg)] = all_candidate_subgroups[best_sg]['modes']
            else:
                break

        print('{} subgroups obtained.'.format(len(sigma)))
        return sigma

def _get_modes_by_prop(g, rl, p):
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

    modes = set()
    for m, events in rl.groupby(['case_type', 'activity_type', 'time_type']):
        candidates_in_group = [
                r for r in events['resource'].drop_duplicates() if r in g]
        if (len(candidates_in_group) > 0 
                and len(candidates_in_group) * 1.0 / len(g) >= p):
            modes.add(m)
    return frozenset(modes)

# Python recipe: this is a helper function
from itertools import chain, combinations
def _powerset_excluded(iterable):
    s = list(iterable)
    return (chain.from_iterable(combinations(s, r) 
        for r in range(1, len(s))))

