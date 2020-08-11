# -*- coding: utf-8 -*-

"""This module contains the implementation of methods for profiling 
resources applying frequent pattern mining techniques.
"""
def from_frequent_patterns(rl):
    """Build resource profiles based on frequent pattern mining, where
    patterns describe resources originating execution modes.
    
    Each column in the result profiles corresponds with an execution
    mode captured in the given resource log.

    Parameters
    ----------
    rl : DataFrame
        A resource log.

    Returns
    -------
    DataFrame
        The constructed resource profiles.
    """
    p = 1 / len(rl)
    all_resources = set(rl['resource'])

    # step 1. build transactions list from resource log
    ld = list()
    for event in rl.itertuples():
        ct = event.case_type if event.case_type != '' else 'CT.'
        at = event.activity_type if event.activity_type != '' else 'AT.'
        tt = event.time_type if event.time_type != '' else 'TT.'

        ld.append([event.resource, ct, at, tt])

    # step 2. discover frequent itemsets
    from pandas import DataFrame
    from mlxtend.preprocessing import TransactionEncoder
    te = TransactionEncoder()
    ld = te.fit(ld).transform(ld)
    ld = DataFrame(ld, columns=te.columns_)

    from mlxtend.frequent_patterns import fpgrowth
    freq_itemsets = fpgrowth(ld, min_support=p, use_colnames=True)

    # step 3. discover association rules
    from mlxtend.frequent_patterns import association_rules
    rules = association_rules(
        freq_itemsets, 
        metric='support',
        min_threshold=p
    )

    # compute evaluation measures
    rules['rev_rule_confidence'] = (
        rules['support'] / rules['consequent support']
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
    rules = rules[
        (rules['antecedents_len'] == 1) &
        (rules['antecedents'].map(
            lambda x: list(x)[0][:3] not in {'CT.', 'AT.', 'TT.'})) &
        (rules['consequents_len'] == 3)
    ]
    rules['antecedents'] = rules['antecedents'].apply(
        lambda x: list(x).pop()
    )
    for row in rules.itertuples():
        if row.wtKulc >= 0.05:
            r = row.antecedents
            all_resources.discard(r)

            rule = [r]
            mode = sorted(
                list(row.consequents),
                key=lambda x: ['CT.', 'AT.', 'TT.'].index(x[:3])
            )
            rule.extend(mode)
            if rule not in results:
                results.append(rule)
        else:
            pass
    
    # append missing resources
    for missing_r in all_resources:
        r_rules = rules.groupby('antecedents').get_group(missing_r)
        r_rules = r_rules.nlargest(1, 'wtKulc')
        for row in r_rules.itertuples():
            mode = sorted(
                list(row.consequents),
                key=lambda x: ['CT.', 'AT.', 'TT.'].index(x[:3])
            )
            results.append([missing_r] + mode)
    # TODO: append missing execution modes?
    
    from collections import defaultdict
    mat = defaultdict(dict)
    for rule in results:
        mode = (
            rule[1] if rule[1].split('.')[1] != '' else '',
            rule[2] if rule[2].split('.')[1] != '' else '',
            rule[3] if rule[3].split('.')[1] != '' else ''
        )
        mat[rule[0]][mode] = True
    
    from pandas import DataFrame
    df = DataFrame.from_dict(mat, orient='index').fillna(False)

    return df
