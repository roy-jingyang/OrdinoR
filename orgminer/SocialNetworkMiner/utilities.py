# -*- coding: utf-8 -*-

'''
This module provides the necessary utilities for preprocessing/processing the
social network mining related data.
'''

def select_edges_by_weight(sn, weight='weight',
        low=None, high=None, percentage=None):
    '''
    This method select and keep edges in a given social network by their weight
    values. The filtering process can be done in either of the two ways:
        1. Define an interval [low, high) and keep only those edges with weight
        values lying within the interval. If a bound is not defined at either
        side, infinity is used instead.
        2. Specify a percentage value and keep only those edges with weight
        values within the range, e.g. with percentage = '+0.75' only edges with
        the top 75% ranked weight values are kept.
    Note that one of the ways must be specified, and the first criteria over-
    rides the second one when both are defined.

    Params:
        sn: NetworkX (Di)Graph
            The mined social network as a NetworkX (Di)Graph object.
        weight: str, optional
            String specifiying the name of the weight attribute in the NetworkX
            graph to be recognized as weight values. The default is 'weight',
            which is the default weight attribute name defined by NetworkX.
        low: float, optional
            The lower bound value, specified when criteria #1 is used.
        high: float, optional
            The upper bound value, specified when criteria #1 is used.
        percentage: str, optional
            The string for specifying percentage value when criteria #2 is
            used, should be formatted as 'sign(+/-)' + 'float in [0, 1)'
    Returns:
        psn: NetworkX (Di)Graph
            The processed social network as a NetworkX (Di)Graph object.
    '''

    from copy import deepcopy
    psn = deepcopy(sn)
    n_edges_old = len(psn.edges)
    edges_to_be_removed = list()
    if low is not None or high is not None:
        test_lower_bound = lambda wt: wt >= low if low is not None else True
        test_upper_bound = lambda wt: wt < high if high is not None else True
        print('\tOnly values within [{}, {}) are kept.'.format(low, high))
        for u, v, wt in psn.edges.data(weight):
            if not (test_lower_bound(wt) and test_upper_bound(wt)):
                edges_to_be_removed.append((u, v))
    elif percentage is not None:
        from numpy import percentile
        if percentage[0] == '+':
            # keep the top p ranked edges
            q = 1 - float(percentage[1:])
            print('\tOnly top {}% are kept.'.format((1 - q) * 100))
            threshold = percentile([e[2] for e in psn.edges.data(weight)], 
                    q * 100)
            test_condition = lambda wt: wt >= threshold
        elif percentage[0] == '-':
            # keep the bottom p ranked edges
            q = float(percentage[1:])
            print('\tOnly bottom {}% are kept.'.format(q * 100))
            threshold = percentile([e[2] for e in psn.edges.data(weight)],
                    q * 100)
            test_condition = lambda wt: wt <= threshold
        else:
            exit('Error processing due to invalid input.')
        
        for u, v, wt in psn.edges.data(weight):
            if not test_condition(wt):
                edges_to_be_removed.append((u, v))
    else:
        exit('Error processing due to no criteria specified.')

    psn.remove_edges_from(edges_to_be_removed)
    print('\t{}/{} edges ({:.2%}) have been removed.'.format(
        len(edges_to_be_removed), n_edges_old,
        (len(edges_to_be_removed) / n_edges_old)))

    return deepcopy(psn)

