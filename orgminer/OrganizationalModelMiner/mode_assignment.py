# -*- coding: utf-8 -*-

"""This module contains methods for determining execution modes 
associated with resource groups.
"""
def full_recall(group, rl):
    """Assign an execution mode to a group, as long as there exists a 
    member resource who has executed the mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    rl : DataFrame
        A resource log.

    Returns
    -------
    modes : list of 3-tuples
        Execution modes associated to the resource group.
    """
    print('Applying FullRecall for mode assignment:')
    modes = list()
    grouped_by_resource = rl.groupby('resource')

    for r in group:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            if m not in modes:
                modes.append(m)
    return modes


def overall_score(group, rl, p, w1=0.5, w2=0.5):
    """Assign an execution mode to a group, as long as the overall score
    (as a weighted average) of its group relative stake and member
    coverage, is higher than a given threshold.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
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
    modes : list of 3-tuples
        Execution modes associated to the resource group, sorted by 
        their relevance in terms of overall score from high to low.
    """
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, member_coverage

    for m in all_execution_modes:
        score = (
            w1 * group_relative_stake(group, m, rl) + 
            w2 * member_coverage(group, m, rl))
        if score >= p:
            tmp_modes.append((m, score))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

