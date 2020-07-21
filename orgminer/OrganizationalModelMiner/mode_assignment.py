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


#NOTE:: Implementation #1 - OverallScore-WA
def overall_score(groups, rl, p=0.5, w1=0.5, w2=None, auto_search=False):
    """Assign an execution mode to a group, as long as the overall score
    (as a weighted average) of its group relative stake and member
    coverage, is higher than a given threshold.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : DataFrame
        A resource log.
    p : float, optional, default 0.5
        A given threshold value in range [0, 1.0].
    w1 : float, optional, default 0.5
        The weight value assigned to participation rate.
    w2 : float, optional, default None
        The weight value assigned to coverage. Note that this parameter
        is in fact redundant as its value must conform with the value set
        for parameter `w1`. See notes below.
    auto_search : bool, optional, default False
        A Boolean flag indicating whether to perform auto grid search to
        determine the threshold and the weightings. When auto grid search
        is required, values in range [0, 1.0] will be tested at a step of 
        0.1, and values given to parameter `p`, `w1`, and `w2` will be
        overridden. Defaults to ``False``, i.e., auto grid search is not 
        to be performed.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from assigning execution modes 
        to resource groups.

    Notes
    -----
        The weighting values are expected to sum up to 1.
    """
    if auto_search is True:
        print('Applying OverallScore with auto search:')
        from orgminer.OrganizationalModelMiner.utilities import grid_search
        from functools import partial
        from copy import deepcopy
        from orgminer.Evaluation.l2m.conformance import f1_score
        solution, params = grid_search(
            partial(overall_score, groups=deepcopy(groups), rl=rl), 
            params_config={
            'p': list(x / 10 for x in range(1, 10)),
            'w1': list(x / 10 for x in range(1, 10))},
            func_eval_score=partial(f1_score, rl)
        )
        print('\tBest solution obtained with parameters:')
        print('\t', end='')
        print(params)
        return solution
    else:
        w2 = 1.0 - w1
        print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
            'and threshold {} '.format(p) + 'for mode assignment:')

    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, group_coverage
    from collections import defaultdict

    scores_group_rel_stake = defaultdict(lambda: defaultdict(dict))
    scores_group_cov = defaultdict(lambda: defaultdict(dict))

    # obtain scores
    for i, group in enumerate(groups):
        for m in all_execution_modes:
            rel_stake = group_relative_stake(group, m, rl)
            scores_group_rel_stake[i][m] = rel_stake
            cov = group_coverage(group, m, rl)
            scores_group_cov[i][m] = cov
    
    # assign based on threshold filtering
    from operator import itemgetter
    om = OrganizationalModel()
    for i, group in enumerate(groups):
        tmp_modes = list()
        for m in all_execution_modes:
            relatedness_score = (
                w1 * scores_group_rel_stake[i][m] +
                w2 * scores_group_cov[i][m]
            )
            if relatedness_score > p:
                tmp_modes.append((m, relatedness_score))
        modes = list(item[0] 
            for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
        om.add_group(group, modes)

    return om


#NOTE: Implementation #2 - OverallScore-HM
def _overall_score(groups, rl, p):
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
    print('Applying OverallScore with ' +
        'threshold {} '.format(p) + 'for mode assignment:')
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, group_coverage
    from collections import defaultdict

    scores_group_rel_stake = defaultdict(lambda: defaultdict(dict))
    min_score_group_rel_stake = 1.0
    max_score_group_rel_stake = 0.0
    scores_group_cov = defaultdict(lambda: defaultdict(dict))
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

            cov = group_coverage(group, m, rl)
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

            if scaled_rel_stake + scaled_cov > 0:
                relatedness_score = (
                    2 * (scaled_rel_stake * scaled_cov) /
                    (scaled_rel_stake + scaled_cov)
                )
                if relatedness_score > p:
                    tmp_modes.append((m, relatedness_score))
            else:
                pass
        modes = list(item[0] 
            for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
        om.add_group(group, modes)

    return om

