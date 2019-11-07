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
    case.

    Params:
        el: DataFrame
            The imported event log.
        normalize: str
            An option for setting the normalization strategy on the edge
            weight. Could be one of the followings:
                - None: no normalization (by default).
                - 'resource': normalized by the amount of cases resources
                  involved in. Note that this may lead to a directed graph
                  derived since the normalization depends on node values.
                - 'total': normalized by the total amount of cases recorded in
                  the event log.
    Returns:
        sn: NetworkX DiGraph
            The mined social network as a NetworkX DiGraph object.
    '''
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
        exit('[Error] Unrecognized option.')

    if is_directed_sn:
        from networkx import DiGraph
        sn = DiGraph(mat)
    else:
        from networkx import Graph
        sn = Graph(mat)
    sn.add_nodes_from(el.groupby('resource').groups.keys()) # include isolates
    return sn

# TODO: developing
def working_similarly(el, case_class_mapping, normalize=None):
    '''
    Params:
        el: DataFrame
            The imported event log.
        case_class_mapping: dict
            The mapping from cases (case ids) to its corresponding case
            class. 
        normalize: str, optional
            An option for setting the normalization strategy on the edge
            weight. Could be one of the followings:
                - None: no normalization (by default).
                - 'resource': normalized by the amount of cases resources
                  involved in. Note that this may lead to a directed graph
                  derived since the normalization depends on node values.
                - 'total': normalized by the total amount of cases recorded in
                  the event log.
    Returns:
        sn: NetworkX DiGraph
            The mined social network as a NetworkX DiGraph object.
    '''
    from collections import defaultdict
    counts_by_class = defaultdict(lambda: defaultdict(lambda: 0))
    for case_id, events in el.groupby('case_id'):
        participants = set(events['resource'])
        case_class = case_class_mapping[case_id]
        for r in participants:
            counts_by_class[r][case_class] += 1
    
    print(counts_by_class)
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    from itertools import combinations
    from scipy.spatial.distance import jaccard
    for pair in combinations(counts_by_class.keys(), r=2):
        u, v = pair[0], pair[1]
        wt = 0.0
        for case_class in case_class_mapping.values():
            if (counts_by_class[u][case_class] > 0 and
                counts_by_class[v][case_class] > 0):
                wt += (counts_by_class[u][case_class] +
                    counts_by_class[v][case_class]) 
            else:
                pass
        mat[u][v]['weight'] = wt

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

