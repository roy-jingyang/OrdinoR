# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on joint cases (ref. van der Aalst et. al, CSCW
2005).
'''

# Working together metric
# TODO: should self loops be considered?
def working_together(el, self_loop=False):
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
        if len(participants) == 1:
            # one resource handling the whole case
            if self_loop:
                r = participants.pop()
                mat[r][r]['weight'] += 1
            else:
                pass
        else:
            # for each pair of participants simultaneously appeared
            for pair in permutations(participants, r=2):
                mat[pair[0]][pair[1]]['weight'] += 1

    # relative notation: divide #joint cases by #cases the resource appeared
    for r, counts in mat.items():
        total_n_cases = sum(d['weight'] for o, d in counts.items()) * 1.0
        for o in counts.keys():
            counts[o]['weight'] /= total_n_cases
    from networkx import DiGraph
    sn = DiGraph(mat)
    sn.add_nodes_from(el.groupby('resource').groups.keys()) # include isolates
    return sn

