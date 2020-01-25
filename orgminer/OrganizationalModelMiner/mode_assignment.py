# -*- coding: utf-8 -*-

"""This module contains methods for determining execution modes 
associated with resource groups.
"""
from orgminer.OrganizationalModelMiner.base import OrganizationalModel

def full_recall(groups, rl):
    """Assign an execution mode to a group, as long as there exists a 
    member resource who has executed the mode.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : DataFrame
        A resource log.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from assigning execution modes 
        to resource groups.
    """
    print('Applying FullRecall for mode assignment:')
    grouped_by_resource = rl.groupby('resource')
    om = OrganizationalModel()

    for group in groups:
        modes = set()
        for r in group:
            # TODO: optimize the update
            modes.update((e.case_type, e.activity_type, e.time_type)
                for e in grouped_by_resource.get_group(r).itertuples())
        om.add_group(group, sorted(list(modes)))
    return om


#TODO: Implementation #1 - OverallScore-WA
def _overall_score(group, rl, p, w1=0.5, w2=0.5):
    """Assign an execution mode to a group, as long as the overall score
    (as a weighted average) of its group relative stake and member
    coverage, is higher than a given threshold.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : DataFrame
        A resource log.
    p : float
        A given threshold value in range [0, 1.0].
    w1 : float, optional, default 0.5
        The weight value assigned to participation rate.
    w2 : float, optional, default 0.5
        The weight value assigned to coverage.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from assigning execution modes 
        to resource groups.
    """
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, member_coverage

    for m in all_execution_modes:
        rel_stake = group_relative_stake(group, m, rl)
        cov = member_coverage(group, m, rl)
        score = w1 * rel_stake + w2 * cov
        if score > p:
            tmp_modes.append((m, score))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes


#TODO: Implementation #2 - OverallScore-HM
def overall_score(group, rl, p):
    """Assign an execution mode to a group, as long as the overall score
    (as a harmonic mean) of its group relative stake and member
    coverage, is higher than a given threshold.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    rl : DataFrame
        A resource log.
    p : float
        A given threshold value in range [0, 1.0].

    Returns
    -------
    modes : list of 3-tuples
        Execution modes associated to the resource group, sorted by 
        their relevance in terms of overall score from high to low.
    """
    print('Applying OverallScore with ' +
        'threshold {} '.format(p) + 'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_focus, group_relative_stake, member_coverage

    for m in all_execution_modes:
        #rel_stake = group_relative_stake(group, m, rl)
        rel_focus = group_relative_focus(group, m, rl)
        cov = member_coverage(group, m, rl)
        #if rel_stake > 0 or cov > 0:
        if rel_focus > 0 or cov > 0:
            #score = 2 * rel_stake * cov / (rel_stake + cov)
            score = 2 * rel_focus * cov / (rel_focus + cov)
            if score > p:
                tmp_modes.append((m, score))
        else:
            pass

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

