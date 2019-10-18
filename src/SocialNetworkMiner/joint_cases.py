# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on joint cases (ref. van der Aalst et. al, CSCW
2005).
'''

# Working together metric
# NOTE: Self loops are not taken into account (those should be 'handovers').
def working_together(el, normalize=None):
    '''
    This method implements the mining based on working together metric, which
    considers how often two individuals are performing activities for the same
    case. Note that a relative notation is used (therefore a directed graph is
    expected to be returned).

    Params:
        el: DataFrame
            The imported event log.
    Returns:
        sn: NetworkX DiGraph
            The mined social network as a NetworkX DiGraph object.
    '''
    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    from itertools import permutations
    for case_id, trace in el.groupby('case_id'):
        participants = set(trace['resource'])
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
        exit('[Error] Unrecognized option.')

    if is_directed_sn:
        from networkx import DiGraph
        sn = DiGraph(mat)
    else:
        from networkx import Graph
        sn = Graph(mat)
    sn.add_nodes_from(el.groupby('resource').groups.keys()) # include isolates
    return sn

