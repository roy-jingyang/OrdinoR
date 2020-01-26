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
def _overall_score(groups, rl, p, w1=0.5, w2=0.5):
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
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, member_coverage
    from collections import defaultdict

    scores_group_rel_stake = defaultdict(lambda: dict)
    min_score_group_rel_stake = 1.0
    max_score_group_rel_stake = 0.0
    scores_group_cov = defaultdict(lambda: dict)
    min_score_group_cov = 1.0
    max_score_group_cov = 0.0

    # obtain scores
    for i, group in enumerate(groups):
        for m in all_execution_modes:
            rel_stake = group_relative_stake(group, m, rl)
            scores_group_rel_stake[i][m] = rel_stake
            min_score_group_rel_stake = rel_stake \
                if rel_stake < min_score_group_rel_stake \
                else min_score_group_rel_stake
            max_score_group_rel_stake = rel_stake \
                if rel_stake > max_score_group_rel_stake \
                else max_score_group_rel_stake

            cov = member_coverage(group, m, rl)
            scores_group_cov[i][m] = cov
            min_score_group_cov = cov \
                if cov < min_score_group_cov \
                else min_score_group_cov
            max_score_group_cov = cov \
                if cov > max_score_group_cov \
                else max_score_group_cov

    def min_max_scale(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    # assign based on threshold filtering
    from operator import itemgetter
    om = OrganizationalModel()
    for i, group in enumerate(groups):
        tmp_modes = list()
        for m in all_execution_modes:
            scaled_rel_stake = min_max_scale(
                scores_group_rel_stake[i][m],
                min_score_group_rel_stake,
                max_score_group_rel_stake)
            scaled_cov = min_max_scale(
                scores_group_cov[i][m],
                min_score_group_cov,
                max_score_group_cov)

            relatedness_score = (
                w1 * scaled_rel_stake + 
                w2 * scaled_cov
            )
            if relatedness_score > p:
                tmp_modes.append((m, relatedness_score))
        modes = list(item[0] 
            for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
        om.add_group(group, modes)

    return om


#TODO: Implementation #2 - OverallScore-HM
def overall_score(group, rl, p):
    """Assign an execution mode to a group, as long as the overall score
    (as a harmonic mean) of its group relative stake and member
    coverage, is higher than a given threshold.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : DataFrame
        A resource log.
    p : float
        A given threshold value in range [0, 1.0].

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from assigning execution modes 
        to resource groups.
    """
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, member_coverage
    from collections import defaultdict

    scores_group_rel_stake = defaultdict(lambda: dict)
    min_score_group_rel_stake = 1.0
    max_score_group_rel_stake = 0.0
    scores_group_cov = defaultdict(lambda: dict)
    min_score_group_cov = 1.0
    max_score_group_cov = 0.0

    # obtain scores
    for i, group in enumerate(groups):
        for m in all_execution_modes:
            rel_stake = group_relative_stake(group, m, rl)
            scores_group_rel_stake[i][m] = rel_stake
            min_score_group_rel_stake = rel_stake \
                if rel_stake < min_score_group_rel_stake \
                else min_score_group_rel_stake
            max_score_group_rel_stake = rel_stake \
                if rel_stake > max_score_group_rel_stake \
                else max_score_group_rel_stake

            cov = member_coverage(group, m, rl)
            scores_group_cov[i][m] = cov
            min_score_group_cov = cov \
                if cov < min_score_group_cov \
                else min_score_group_cov
            max_score_group_cov = cov \
                if cov > max_score_group_cov \
                else max_score_group_cov

    def min_max_scale(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    # assign based on threshold filtering
    from operator import itemgetter
    om = OrganizationalModel()
    for i, group in enumerate(groups):
        tmp_modes = list()
        for m in all_execution_modes:
            scaled_rel_stake = min_max_scale(
                scores_group_rel_stake[i][m],
                min_score_group_rel_stake,
                max_score_group_rel_stake)
            scaled_cov = min_max_scale(
                scores_group_cov[i][m],
                min_score_group_cov,
                max_score_group_cov)

            relatedness_score = (
                2 * (scaled_rel_stake * scaled_cov) /
                (scaled_rel_stake + scaled_cov)
            )
            if relatedness_score > p:
                tmp_modes.append((m, relatedness_score))
        modes = list(item[0] 
            for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
        om.add_group(group, modes)

    return om

