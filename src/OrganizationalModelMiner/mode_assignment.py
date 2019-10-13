# -*- coding: utf-8 -*-

'''
This module contains methods for associating discovered organizational groups
with execution modes.
'''

'''
The following (private) methods are used for assessing the relatedness between
a group and an execution mode regarding different factors (perspectives).
'''
def _participation_rate(group, mode, rl):
    '''Measure the participation rate of a group with respect to an execution
    mode.

    Params:
        group: iterator
            The ids of resources as a resource group.
        mode: tuple
            The execution mode.
        rl: DataFrame
            The resource log.
    Returns:
        : float
            The participated rate measured.
    '''
    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events

    total_par_count = len(rl)
    par_count = len(
        rl.loc[
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]
        )
            
    return par_count / total_par_count

def _coverage(group, mode, rl):
    '''Measure the coverage of a group with respect to an execution mode.

    Params:
        group: iterator
            The ids of resources as a resource group.
        mode: tuple
            The execution mode.
        rl: DataFrame
            The resource log.
    Returns:
        : float
            The coverage measured.
    '''
    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events

    num_participants = 0
    for r in group:
        if len(rl.loc[
            (rl['resource'] == r) &
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]) > 0:
            num_participants += 1
        else:
            pass
    
    return num_participants / len(group)

'''
The following methods are for determining a set of execution modes for a given
group.
'''
def full_recall(group, rl):
    '''Assign an execution mode to a group, as long as there exists a member
    resource of this group that have executed this mode, i.e. everything done
    by each of the members matters.

    Note: this is the method proposed by Song & van der Aalst, DSS 2008,
    namely "entity_assignment".

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
    Returns:
        modes: frozenset
            The execution modes corresponded to the resources.
    '''
    print('Applying FullRecall for mode assignment:')
    modes = set()
    grouped_by_resource = rl.groupby('resource')

    for r in group:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            modes.add(m)

    return frozenset(modes)

def participation_first(group, rl, p):
    '''
    '''
    print('Applying ParticipationFirst with threshold {} '.format(p) +
        'for mode assignment:')
    modes = set()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        if _participation_rate(group, m, rl) >= p:
            modes.add(m)

    return frozenset(modes)

def coverage_first(group, rl, p):
    '''
    '''
    print('Applying CoverageFirst with threshold {} '.format(p) +
        'for mode assignment:')
    modes = set()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        if _coverage(group, m, rl) >= p:
            modes.add(m)

    return frozenset(modes)

def overall_score(group, rl, p, w1=0.5, w2=0.5):
    '''
    '''
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    modes = set()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        if (w1 * _participation_rate(group, m, rl) + 
            w2 * _coverage(group, m, rl)) >= p:
            modes.add(m)

    return frozenset(modes)

# [DEPRECATED]
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
            The designated proportion value.
    Returns:
        modes: frozenset/dict of frozensets
            The execution modes corresponded to the resources:
            - if a splitting is not needed, then a frozenset is returned;
            - if a splitting is needed, then a dict of frozensets is returned,
              with the subgroups as dict keys and the related execution modes
              as values.
    '''
    print('Applying "assign by proportion {:.0%}" for mode assignment:'.format(
        p))

    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events
    # pre-computing {mode -> candidate resources} (inv. of Resource Capability)
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

def assign_by_weighting(group, rl, profiles, metric='euclidean'):
    '''Assign execution modes to a group based on how each member resource of
    this group contribute the clustering effectiveness of this group. Only 
    cohesion is considered. 
    Member resources with relatively higher contribution are recognized as 
    representatives of the group, and by whom the execution modes executed will
    be assigned to the group.

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation. This should be consistent with that employed within
            the mining method.
    Returns:
        modes: frozenset
            The execution modes corresponded to the resources.
    '''
    print('Applying "assign by weighting" for mode assignment:')
    from numpy import mean, amin
    from math import ceil
    from scipy.spatial.distance import cdist, pdist

    # pre-computing {resource -> modes) (Resource Capability)
    grouped_by_resource = rl.groupby('resource')
    resource_cap = dict()
    for r in group:
        resource_cap[r] = frozenset(
                (e.case_type, e.activity_type, e.time_type) for e in 
                grouped_by_resource.get_group(r).itertuples())

    if len(group) == 1:
        representatives = list(group)
    else:
        # find the representatives
        representatives = list()
        # calculate the contribution score (use Local Outlier Factor here)
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(metric=metric)

        group = sorted(list(group)) # NOTE: order preserved for enumeration
        lof.fit(profiles.loc[group])
        r_scores = list()
        for i, r in enumerate(group):
            r_scores.append((r, lof.negative_outlier_factor_[i]))

        # select the representative(s) with the maximal score
        # NOTE: there may be more than one object with the maximal score!
        representatives = list()
        maximal_score = float('-inf')
        for x in r_scores:
            if x[1] >= maximal_score:
                if x[1] > maximal_score:
                    representatives.clear()
                    maximal_score = x[1]
                representatives.append(x[0])
            else:
                pass
        

        # TODO: could extend to a top-k solution (see below)
        '''
        r_scores.sort(key=lambda x: x[1], reverse=True)
        #ratio_representatives = 0.1
        #n_representatives = ceil(len(group) * ratio_representatives)

        # select the representative(s) with top-k best scores
        representatives = [x[0] for x in r_scores[:n_representatives]]
        '''
        
    # determine the modes to be assigned based on the representatives
    modes = frozenset.union(
            *list(resource_cap[r] for r in representatives))
    return modes

