"""
Construct an organizational model by profiling resource groups, given
groups and an event log (or the corresponding resource log)
"""

from functools import partial
from copy import deepcopy
from collections import defaultdict
from operator import itemgetter

from ordinor.conformance import f1_score
from ordinor.analysis.group_profiles import \
    group_relative_stake as relative_stake, \
    group_coverage as coverage
import ordinor.constants as const

from .models.base import OrganizationalModel
from ._helpers import grid_search

def full_recall(groups, rl):
    """
    Assign an execution context to a group, as long as there exists one
    member resource who has executed the context.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : pandas.DataFrame
        A resource log.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from profiling resource groups
        with execution contexts.
    """
    print('Applying FullRecall for profiling groups:')
    grouped_by_resource = rl.groupby(const.RESOURCE)
    om = OrganizationalModel()

    for group in groups:
        caps = set()
        for r in group:
            # TODO: optimize the update
            caps.update(
                (e[const.CASE_TYPE], 
                 e[const.ACTIVITY_TYPE], 
                 e[const.TIME_TYPE]
                )
                for e in grouped_by_resource.get_group(r).to_dict(orient='records')
            )
        om.add_group(group, sorted(caps))
    return om


def overall_score(groups, rl, p=0.5, w1=0.5, w2=None, auto_search=False):
    """
    Assign an execution context to a group, as long as the overall 
    score (as a weighted average) of its group relative stake and member
    coverage is higher than a given threshold.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : pandas.DataFrame
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
        An organizational model resulted from profiling resource groups
        with execution contexts.

    Notes
    -----
        The weighting values are expected to sum up to 1.
    """
    if auto_search is True:
        print('Applying OverallScore with auto search:')
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
            'and threshold {} '.format(p) + 'for profiling groups:')

    all_exe_ctxs = set(
        (re[const.CASE_TYPE], 
         re[const.ACTIVITY_TYPE], 
         re[const.TIME_TYPE]
        ) 
        for re in rl.drop_duplicates().to_dict(orient='records')
    )

    scores_group_rel_stake = defaultdict(lambda: defaultdict(dict))
    scores_group_cov = defaultdict(lambda: defaultdict(dict))

    # obtain scores
    for i, group in enumerate(groups):
        for m in all_exe_ctxs:
            rel_stake = relative_stake(group, m, rl)
            scores_group_rel_stake[i][m] = rel_stake
            cov = coverage(group, m, rl)
            scores_group_cov[i][m] = cov
    
    # assign based on threshold filtering
    om = OrganizationalModel()
    for i, group in enumerate(groups):
        tmp_ctxs = list()
        for m in all_exe_ctxs:
            relatedness_score = (
                w1 * scores_group_rel_stake[i][m] +
                w2 * scores_group_cov[i][m]
            )
            if relatedness_score > p:
                tmp_ctxs.append((m, relatedness_score))
        caps = list(item[0] 
            for item in sorted(tmp_ctxs, key=itemgetter(1), reverse=True)
        )
        om.add_group(group, caps)

    return om
