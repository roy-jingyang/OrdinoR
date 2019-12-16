# -*- coding: utf-8 -*-

"""This module provides the necessary utilities for processing data 
structures related to social network mining.
"""
def select_edges_by_weight(sn, weight='weight',
    low=None, high=None, 
    percentage=None):
    '''Select and keep edges in an input social network based on edge 
    weight values. 
    
    Notes
    -----
    The filtering process can be done in either of the two ways:
        
        1. Define an interval [`low`, `high`) and keep only edges 
        with weight values falling within the interval. If either one of 
        the two sides of bound is not defined, infinity is used instead.
        
        2. Specify a percentage value and keep only edges with weight
        values within the range, e.g., with ``'+0.75'`` specified, only 
        edges with the highest 75% weight values are kept.

    Note that one of the two ways must be specified, and the first would 
    override the second if both are presented.

    Parameters
    ----------
    sn : NetworkX (Di)Graph
        A social network.
    weight : str, optional, default 'weight'
        Name of the weight attribute in the network. Defaults to 
        ``'weight'``, which is the default weight attribute name used by 
        NetworkX.
    low : float, optional
        The lower bound value, used for the first way of filtering.
    high : float, optional
        The upper bound value, used for the first way of filtering.
    percentage : str, optional
        The specified percentage value, used for the second way of 
        filtering. Should be formatted as a plus/minus sign concatenated 
        with a float number in range [0, 1), e.g., ``+0.75``.

    Returns
    -------
    psn: NetworkX (Di)Graph
        The social network with certain edges filtered out.
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
            threshold = percentile([e[2] 
                for e in psn.edges.data(weight)], q * 100)
            test_condition = lambda wt: wt >= threshold
        elif percentage[0] == '-':
            # keep the bottom p ranked edges
            q = float(percentage[1:])
            print('\tOnly bottom {}% are kept.'.format(q * 100))
            threshold = percentile([e[2] 
                for e in psn.edges.data(weight)], q * 100)
            test_condition = lambda wt: wt <= threshold
        else:
            exit('Error processing due to invalid input.')
        
        for u, v, wt in psn.edges.data(weight):
            if not test_condition(wt):
                edges_to_be_removed.append((u, v))
    else:
        raise RuntimeError('No filtering criterion specified.')

    psn.remove_edges_from(edges_to_be_removed)
    print('\t{}/{} edges ({:.2%}) have been removed.'.format(
        len(edges_to_be_removed), n_edges_old,
        len(edges_to_be_removed) / n_edges_old))

    return deepcopy(psn)

