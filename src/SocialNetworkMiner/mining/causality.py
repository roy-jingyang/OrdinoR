#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on (possible) causality (ref. van der Aalst et.
al, CSCW 2005).

Warning: Including real causality into consideration requires support of a
process model (related to the event log). Ignore this option for now.
'''

# Handover of work metric
def handover(c,
        real_causality, direct_succession, multiple_transfers,
        depth=1, beta=1):
    '''
    This method implements the mining based on handover of work metric, which
    considers the relationship between a pair of resources completing sub-
    sequent activities in process execution.

    Params:
        c: DataFrame
            The imported event log.
        real_causality: boolean
            Consider arbitrary transfers of work OR only those where there is a
            causal dependency (determined by the process model).
        direct_succession: boolean
            Consider the degree of causality - direct OR indirect succession.
        multiple_transfers: boolean
            Consider multiple transfers within one instance (case) OR not.
        depth: int, optional
            The degree of causality specified. Only use when
            'direct_succession' is False, i.e. indirect succession considered.
        beta: float, in range [0, 1], optional
            The causality fall factor.
    Returns:
        sn: NetworkX DiGraph
            The mined social network as a NetworkX DiGraph object.
    '''

    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    if direct_succession and multiple_transfers: # CDCM
        # scale_factor: SIGMA_Case c in Log (|c| - 1)
        sf = sum(len(trace) - 1 for case_id, trace in c.groupby('case_id'))
        for case_id, trace in c.groupby('case_id'):
            for i in range(len(trace) - 1):
                res_prev = trace.iloc[i]['resource']
                res_next = trace.iloc[i + 1]['resource']
                if res_prev != res_next: # self-loop ignored
                    mat[res_prev][res_next]['weight'] += 1 / sf

    elif direct_succession and not multiple_transfers: # CDIM
        # scale_factor: |L|
        sf = len(c.groupby('case_id'))
        for case_id, trace in c.groupby('case_id'):
            handovered_pairs = set()
            for i in range(len(trace) - 1):
                res_prev = trace.iloc[i]['resource']
                res_next = trace.iloc[i + 1]['resource']
                if res_prev != res_next: # self-loop ignored
                    handovered_pairs.add((res_prev, res_next))
            for pair in handovered_pairs:
                mat[pair[0]][pair[1]]['weight'] += 1 / sf

    elif not direct_succession and multiple_transfers: # CICM 
        # scale_factor: SIGMA_Case c in Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1 * (|c| - n)))
        sf = 0
        for case_id, trace in c.groupby('case_id'):
            num_events = len(trace)
            sf += sum(beta ** (n - 1) * (num_events - n) \
                    for n in range(1, min(num_events - 1, depth)))
        for case_id, trace in c.groupby('case_id'):
            # for each case
            num_events = len(trace)
            for i in range(num_events - 1):
                for n in range(1, min(num_events - 1, depth)):
                    # for each level of calculation depth
                    if i + n <= num_events - 1:
                        res_prev = trace.iloc[i]['resource']
                        res_next = trace.iloc[i + 1]['resource']
                        if res_prev != res_next: # self-loop ignored
                            mat[res_prev][res_next]['weight'] += \
                                    beta ** (n - 1) * 1 / sf

    else: # CIIM
        # scale_factor: SIGMA_Case c in Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1))
        sf = 0
        for case_id, trace in c.groupby('case_id'):
            num_events = len(trace)
            sf += sum(beta ** (n - 1) \
                    for n in range(1, min(num_events - 1, depth)))
        for case_id, trace in c.groupby('case_id'):
            # for each case
            handovered_pairs = set()
            num_events = len(trace)
            for i in range(num_events - 1):
                for n in range(1, min(num_events - 1, depth)):
                    # for each level of calculation depth
                    if i + n <= num_events - 1:
                        res_prev = trace.iloc[i]['resource']
                        res_next = trace.iloc[i + 1]['resource']
                        if res_prev != res_next: # self-loop ignored
                            handovered_pairs.add((res_prev, res_next, 
                                beta ** (n - 1)))
            for pair in handovered_pairs:
                mat[pair[0]][pair[1]]['weight'] += pair[2] * 1 / sf

    from networkx import DiGraph
    sn = DiGraph(mat)
    sn.add_nodes_from(c.groupby('resource').groups.keys())
    return sn

# TODO
# Subcontracting metric
def subcontracting():
    pass

