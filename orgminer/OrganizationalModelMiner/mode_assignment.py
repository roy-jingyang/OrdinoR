# -*- coding: utf-8 -*-

"""This module contains methods for determining execution modes 
associated with resource groups.
"""
def _participation_rate(group, mode, rl):
    """Measure the participation rate of a group with respect to an 
    execution mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    mode : 3-tuple
        An execution mode.
    rl : DataFrame
        A resource log.

    Returns
    -------
    float
        The measured participated rate.
    """
    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)]

    total_par_count = len(rl)
    par_count = len(
        rl.loc[
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]
    )
    return par_count / total_par_count


def _coverage(group, mode, rl):
    """Measure the coverage of a group with respect to an execution mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    mode : 3-tuple
        An execution mode.
    rl : DataFrame
        A resource log.

    Returns
    -------
    float
        The measured coverage.
    """
    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)]

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


def participation_first(group, rl, p):
    """Assign an execution mode to a group, as long as its participation 
    rate is higher than a given threshold.

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
        their relevance in terms of participation rate from high to low.
    """
    print('Applying ParticipationFirst with threshold {} '.format(p) +
        'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        par_rate = _participation_rate(group, m, rl)
        if par_rate >= p:
            tmp_modes.append((m, par_rate))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes


def coverage_first(group, rl, p):
    """Assign an execution mode to a group, as long as its coverage is
    higher than a given threshold.

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
        their relevance in terms of coverage from high to low.
    """
    print('Applying CoverageFirst with threshold {} '.format(p) +
        'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        coverage = _coverage(group, m, rl)
        if coverage >= p:
            tmp_modes.append((m, coverage))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes


def overall_score(group, rl, p, w1=0.5, w2=0.5):
    """Assign an execution mode to a group, as long as its overall score
    of considering participation and coverage simultaneously, is higher 
    than a given threshold.

    The overall score is calculated as a weighted average number of the 
    two measures.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    rl : DataFrame
        A resource log.
    p : float
        A given threshold value in range [0, 1.0].
    w1 : float, optional
        The weight value assigned to participation rate, default 0.5.
    w2 : float, optional
        The weight value assigned to coverage, default 0.5.

    Returns
    -------
    modes : list of 3-tuples
        Execution modes associated to the resource group, sorted by 
        their relevance in terms of overall score from high to low.

    See Also
    --------
    participation_first
    coverage_first
    """
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        score = (w1 * _participation_rate(group, m, rl) + 
            w2 * _coverage(group, m, rl))
        if score >= p:
            tmp_modes.append((m, score))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

