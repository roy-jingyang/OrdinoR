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
        An organizational model resulted profiling resource groups with 
        execution modes.
    """
    print('Applying FullRecall for group profiling:')
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
        A boolean flag indicating whether to perform auto grid search to
        determine the threshold and the weightings. When auto grid search
        is required, values in range [0, 1.0] will be tested at a step of 
        0.1, and values given to parameter `p`, `w1`, and `w2` will be
        overridden. Defaults to ``False``, i.e., auto grid search is not 
        to be performed.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted profiling resource groups with 
        execution modes.

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
            'and threshold {} '.format(p) + 'for group profiling:')

    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_stake, group_coverage
    from collections import defaultdict

    #scores_group_rel_focus = defaultdict(lambda: defaultdict(dict))
    scores_group_rel_stake = defaultdict(lambda: defaultdict(dict))
    scores_group_cov = defaultdict(lambda: defaultdict(dict))

    # obtain scores
    for i, group in enumerate(groups):
        for m in all_execution_modes:
            #rel_focus = group_relative_focus(group, m, rl)
            #scores_group_rel_focus[i][m] = rel_focus
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
        An organizational model resulted profiling resource groups with 
        execution modes.
    """
    print('Applying OverallScore with ' +
        'threshold {} '.format(p) + 'for group profiling:')
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


# TODO: params {p, threshold_wtKulc, threshold_coverage}
def association_rules(groups, rl, p=None):
    """Profile resources groups using association rule mining.

    Parameters
    ----------
    groups : list of sets
        Resource groups containing resource ids.
    rl : DataFrame
        A resource log.
    p : float, optional, default None
        A given threshold value in range [0, 1.0] for performing frequent
        itemsets discovery, i.e., ``min_support``. If not provided, it
        will be set to a low value such that all execution modes with
        occurrence in the log will be considered for group profiling.

    Returns
    -------
    om : OrganizationalModel
        An organizational model resulted from assigning execution modes 
        to resource groups.
    """
    print('Applying AssociationRule for group profiling:')

    # initialize p with a lowest possible value if not specified
    p = 1 / len(rl) if p is None else p

    # step 1. build membership mapping
    membership = dict()
    for i, group in enumerate(groups):
        for r in group:
            if r in membership:
                # TODO: determine solution for the overlapping case
                raise NotImplementedError
            membership[r] = 'RG.{}'.format(i)
    
    # step 2. build transactions list from resource log
    ld = list()
    # also store the coverage information
    from collections import defaultdict
    group_cov_count = defaultdict(lambda: defaultdict(set))
    group_cov = defaultdict(dict)

    for event in rl.itertuples():
        group_label = membership[event.resource]
        ct = event.case_type if event.case_type != '' else 'CT.'
        at = event.activity_type if event.activity_type != '' else 'AT.'
        tt = event.time_type if event.time_type != '' else 'TT.'

        ld.append([group_label, ct, at, tt])
        group_cov_count[group_label][(ct, at, tt)].add(event.resource)
    
    for group_label, v in group_cov_count.items():
        group_index = int(group_label[3:])
        for mode in v.keys():
            group_cov[group_label][mode] = (
                len(group_cov_count[group_label][mode]) /
                len(groups[group_index])
            )
    
    # step 3. discover frequent itemsets
    from pandas import DataFrame
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    ld = te.fit(ld).transform(ld)
    ld = DataFrame(ld, columns=te.columns_)

    from mlxtend.frequent_patterns import fpgrowth
    freq_itemsets = fpgrowth(ld, min_support=p, use_colnames=True)

    # step 4. discover association rules
    from mlxtend.frequent_patterns import association_rules
    rules = association_rules(
        freq_itemsets, 
        metric='support',
        min_threshold=p
    )

    # compute additional measures
    rules['rev_rule_confidence'] = (
        rules['support'] / rules['consequent support']
    )
    # max confidence: max. of the confidence values
    rules['max_confidence'] = rules[
        ['confidence', 'rev_rule_confidence']
    ].max(axis=1)
    # Kulcyznski measure: average of the confidence values
    rules['Kulc'] = 0.5 * (rules['confidence'] + rules['rev_rule_confidence'])
    # cosine measure: (geo mean of the confidence values)
    rules['cosine'] = ((rules['confidence'] * rules['rev_rule_confidence']) 
        ** 0.5)
    # Imbalance Ratio
    rules['IR'] = (
        abs(rules['antecedent support'] - rules['consequent support']) /
        (rules['antecedent support'] + rules['consequent support'] -
         rules['support'])
    )
    # IW: imbalance weighting, derived from Imbalance Ratio value
    rules['IW'] = (1 +
        (rules['antecedent support'] - rules['consequent support']) /
        (rules['antecedent support'] + rules['consequent support'] -
         rules['support'])
    )
    # wtKulc: weighted Kulc measure, combined with calculation of IR
    rules['wtKulc'] = (0.5 * (
        (2 - rules['IW']) * rules['confidence'] +
        rules['IW'] * rules['rev_rule_confidence']
    ))

    rules["antecedents_len"] = rules["antecedents"].apply(len)
    rules["consequents_len"] = rules["consequents"].apply(len)

    # filter derived rules
    results = list()
    # Criteria:
    # C1. rules must be in the form of "Group (len==1) => Mode (len==3)"
    # C2. at least one rule must be retained for each of the groups
    # C3. rules must have decent relative focus and relative stake
    # C4. rare but interesting rules (patterns):
    #   having decent coverage
    rules = rules[
        (rules['antecedents_len'] == 1) &
        (rules['antecedents'].map(lambda x: list(x)[0].startswith('RG.'))) &
        (rules['consequents_len'] == 3)
    ]
    for v_group, cand_rules in rules.groupby('antecedents'):
        filters = cand_rules['max_confidence'] >= 0.1
        filtered_cand_rules = cand_rules[filters]
        if len(filtered_cand_rules) == 0:
            filtered_cand_rules = cand_rules.nlargest(1, 'max_confidence')

        for row in filtered_cand_rules.itertuples():
            rule = sorted(
                frozenset.union(row.antecedents, row.consequents), 
                key=lambda x: ['RG', 'CT', 'AT', 'TT'].index(x[:2])
            )
            if rule not in results:
                results.append(tuple(rule))
        
        # "recycle" rules by checking group coverage
        for row in cand_rules[~filters].itertuples():
            rule = sorted(
                frozenset.union(row.antecedents, row.consequents), 
                key=lambda x: ['RG', 'CT', 'AT', 'TT'].index(x[:2])
            )
            if rule not in results:
                if group_cov[rule[0]][tuple(rule[1:])] > 0.5:
                    results.append(tuple(rule))

    # step 5. use association rules for group profiling
    om = OrganizationalModel()
    from collections import defaultdict
    group_caps = defaultdict(list)
    for rule in results:
        group_index = int(rule[0][3:])
        mode = (
            rule[1] if rule[1].split('.')[1] != '' else '',
            rule[2] if rule[2].split('.')[1] != '' else '',
            rule[3] if rule[3].split('.')[1] != '' else ''
        )
        group_caps[group_index].append(mode)
    
    for i, group in enumerate(groups):
        om.add_group(group, group_caps[i])

    return om
