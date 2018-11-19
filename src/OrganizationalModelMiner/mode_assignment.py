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
        # post refinement required
        # TODO: different cost function definitions
        def cost(g):
            m = _get_modes_by_prop(frozenset(g), inv_resource_cap, p)
            if len(m) >= 1:
                #return 1.0 # Naive
                #return 1.0 / len(g) # Max. Size
                #return len(g) # Min. Size
                return 1.0 / len(m) # Max. Cap.
            else:
                return float('inf')

        #search_option = 'exhaust'
        # NOTE: pruning only applies for Naive and Max. Size
        #search_option = 'pruned'
        search_option = 'ga'
        split = _set_cover_greedy(group, cost, search=search_option)

        print('\t[Warning] Group of size {} split to {} subgroups, '
              'after refinement using search method "{}".'
            .format(len(group), len(split), search_option))
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

def _set_cover_greedy(U, f_cost, search='exhaust'):
    from .utilities import powerset_exclude_headtail
    sigma = list()

    while True:
        if len(sigma) > 0:
            uncovered = U.difference(frozenset.union(*sigma))
        else:
            uncovered = U
        print('\t\t{}/{} uncovered.'.format(len(uncovered), len(U)))

        if len(uncovered) > 0:
            def cost_effectiveness(g):
                cost = f_cost(g)
                if cost != float('inf'):
                    # valid candidate
                    return len(uncovered.intersection(frozenset(g))) / cost
                else:
                    # invalid candidate
                    return float('-inf')
            
            best_candidate = None
            if search == 'exhaust':
                candidates = powerset_exclude_headtail(
                        sorted(U), reverse=True)
                best_candidate = max(candidates, key=cost_effectiveness)
            elif search == 'pruned':
                def find_best_candidate(S):
                    best = None
                    best_coverable = None
                    best_cost_effectiveness = float('-inf')
                    best_coverablity = 0

                    for cand in S:
                        num_covered = len(uncovered.intersection(frozenset(cand)))
                        if num_covered > best_coverablity:
                            best_coverablity = num_covered
                            best_coverable = cand

                        c_e = cost_effectiveness(cand)
                        if c_e > 0 and c_e > best_cost_effectiveness:
                            best_cost_effectiveness = c_e
                            best = cand

                    if best is not None:
                        return best
                    else:
                        return find_best_candidate(powerset_exclude_headtail(
                            best_coverable, reverse=True, depth=1))

                candidates = powerset_exclude_headtail(
                        sorted(U), reverse=True, depth=1)
                best_candidate = find_best_candidate(candidates)
            elif search == 'ga':
                from .utilities import find_best_subset_GA
                sorted_U = sorted(U)
                def f_evaluate(individual):
                    s = set(sorted_U[i]
                            for i, flag in enumerate(individual) if flag == 1)
                    return (cost_effectiveness(s),)
                from random import random
                GA_SIZE_POP = 100
                GA_PR_CX = 0.5 # probability of crossover
                GA_PR_MT = 0.2 # probability of mutation
                
                while True:
                    result = find_best_subset_GA(sorted_U,
                            evaluate=f_evaluate,
                            seed=uncovered,
                            max_iter=500,
                            size_population=min(
                                (2 ** len(sorted_U) - 2), GA_SIZE_POP), 
                            p_crossover=GA_PR_CX, p_mutate=GA_PR_MT)
                    if cost_effectiveness(result) > 0:
                        best_candidate = result
                        break
                    else:
                        # TODO: add randomness here
                        pass
            else:
                exit('[Error] Invalid option specified for search method.')
            
            if best_candidate is not None:
                sigma.append(frozenset(best_candidate))
            else:
                exit('[Fatal Error] No valid search result produced')
        else:
            return sigma

