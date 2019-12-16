# -*- coding: utf-8 -*-

"""This module contains the implementation of methods for mining social 
networks from an event log, using metrics based on joint cases [1]_.

See Also
--------
orgminer.SocialNetworkMiner.causality
orgminer.SocialNetworkMiner.joint_activities

References
----------
.. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
   Discovering social networks from event logs. *Computer Supported
   Cooperative Work (CSCW)*, 14(6), 549-593.
   `<https://doi.org/10.1007/s10606-005-9005-9>`_
"""
def working_together(el, normalize=None):
    """Discover a social network from an event log based on working 
    together metric, considering how often two resources were involved 
    in the same case.

    Parameters
    ----------
    el : DataFrame
        An event log.
    normalize : {None, 'resource', 'total'}, optional, default None
        Options for setting the normalization strategy on the edge
        weight values. Could be one of the following:

            - ``None``, no normalization will be used.
            - ``'resource'``, normalized by the amount of cases each 
              resource was involved in. Note that this could lead to a 
              directed graph being derived since the normalization 
              is subject to each individual resource.
            - ``'total'``, normalized by the total amount of cases 
              recorded in the event log.

    Returns
    -------
    sn : NetworkX Graph or DiGraph
        The discovered social network.
    
    Raises
    ------
    ValueError
        If the specified option for normalization is invalid.
    """
    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    from itertools import permutations
    for case_id, events in el.groupby('case_id'):
        participants = set(events['resource'])
        # for each pair of participants simultaneously appeared
        for pair in permutations(participants, r=2):
            mat[pair[0]][pair[1]]['weight'] += 1

    if normalize is None:
        is_directed_sn = False
    elif normalize == 'resource':
        is_directed_sn = True
        # count for number of cases a resource participated
        res_case_count = defaultdict(lambda: 0)
        for res_case, events in el.groupby(['resource', 'case_id']):
            res_case_count[res_case[0]] += 1
        for r, counts in mat.items():
            for o in counts.keys():
                counts[o]['weight'] /= res_case_count[r]
    elif normalize == 'total':
        is_directed_sn = False
        total_num_cases = len(set(el['case_id']))
        for r, counts in mat.items():
            for o in counts.keys():
                counts[o]['weight'] /= total_num_cases
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'normalize', normalize))

    if is_directed_sn:
        from networkx import DiGraph
        sn = DiGraph(mat)
    else:
        from networkx import Graph
        sn = Graph(mat)

    # include isolates
    sn.add_nodes_from(el.groupby('resource').groups.keys())
    return sn

