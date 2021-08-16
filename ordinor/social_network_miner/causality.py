"""
Mining social networks from an event log using metrics based on causality
[1]_

See Also
--------
ordinor.social_network_miner.joint_activities
ordinor.social_network_miner.joint_cases

Notes
-----
Considering real causality requires the presence of a process model
related to the given event log.

References
----------
.. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
   Discovering social networks from event logs. *Computer Supported
   Cooperative Work (CSCW)*, 14(6), 549-593.
   `<https://doi.org/10.1007/s10606-005-9005-9>`_
"""

from collections import defaultdict

from networkx import DiGraph

import ordinor.constants as const
from ordinor.utils.validation import check_convert_input_log

def handover(el,
    real_causality, direct_succession, multiple_transfers,
    depth=1, beta=1):
    """
    Discover a social network from an event log based on handover of 
    work metric, which considers the relationship between a pair of 
    resources completing subsequent activities in process execution.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log.
    real_causality : bool
        A Boolean flag indicating whether to consider arbitrary 
        transfers of work, or only those for which there is a causal 
        dependency according to the process model.
    direct_succession : bool
        A Boolean flag indicating whether to consider the degree of 
        causality, i.e., direct or indirect succession.
    multiple_transfers : bool
        A Boolean flag indicating whether to consider multiple transfers 
        within one case or not.
    depth : int, optional, default 1
        The degree of causality to be considered. Only use when 
        `direct_succession` is False, i.e., indirect succession 
        considered.
    beta : float, optional, default 1
        The causality fall factor.

    Returns
    -------
    sn : NetworkX DiGraph
        The discovered social network.
    """
    el = check_convert_input_log(el)
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    if direct_succession and multiple_transfers: # CDCM
        # scale_factor: SIGMA_Case c in Log (|c| - 1)
        sf = sum(len(trace) - 1 for case_id, trace in el.groupby(const.CASE_ID))
        for case_id, trace in el.groupby(const.CASE_ID):
            for i in range(len(trace) - 1):
                res_prev = trace.iloc[i][const.RESOURCE]
                res_next = trace.iloc[i + 1][const.RESOURCE]
                if res_prev != res_next: # self-loop ignored
                    mat[res_prev][res_next]['weight'] += 1 / sf

    elif direct_succession and not multiple_transfers: # CDIM
        # scale_factor: |L|
        sf = len(el.groupby(const.CASE_ID))
        for case_id, trace in el.groupby(const.CASE_ID):
            handovered_pairs = set()
            for i in range(len(trace) - 1):
                res_prev = trace.iloc[i][const.RESOURCE]
                res_next = trace.iloc[i + 1][const.RESOURCE]
                if res_prev != res_next: # self-loop ignored
                    handovered_pairs.add((res_prev, res_next))
            for pair in handovered_pairs:
                mat[pair[0]][pair[1]]['weight'] += 1 / sf

    elif not direct_succession and multiple_transfers: # CICM 
        # scale_factor: SIGMA_Case c in 
        # Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1 * (|c| - n)))
        sf = 0
        for case_id, trace in el.groupby(const.CASE_ID):
            num_events = len(trace)
            sf += (sum(beta ** (n - 1) * (num_events - n) 
                for n in range(1, min(num_events - 1, depth))))
        for case_id, trace in el.groupby(const.CASE_ID):
            # for each case
            num_events = len(trace)
            for i in range(num_events - 1):
                for n in range(1, min(num_events - 1, depth)):
                    # for each level of calculation depth
                    if i + n <= num_events - 1:
                        res_prev = trace.iloc[i][const.RESOURCE]
                        res_next = trace.iloc[i + 1][const.RESOURCE]
                        if res_prev != res_next: # self-loop ignored
                            mat[res_prev][res_next]['weight'] += (
                                beta ** (n - 1) * 1 / sf)

    else: # CIIM
        # scale_factor: SIGMA_Case c in 
        # Log (SIGMA_n=1:min(|c| - 1, depth) (beta^n-1))
        sf = 0
        for case_id, trace in el.groupby(const.CASE_ID):
            num_events = len(trace)
            sf += (sum(beta ** (n - 1) 
                for n in range(1, min(num_events - 1, depth))))
        for case_id, trace in el.groupby(const.CASE_ID):
            # for each case
            handovered_pairs = set()
            num_events = len(trace)
            for i in range(num_events - 1):
                for n in range(1, min(num_events - 1, depth)):
                    # for each level of calculation depth
                    if i + n <= num_events - 1:
                        res_prev = trace.iloc[i][const.RESOURCE]
                        res_next = trace.iloc[i + 1][const.RESOURCE]
                        if res_prev != res_next: # self-loop ignored
                            handovered_pairs.add((
                                res_prev, res_next, 
                                beta ** (n - 1)
                            ))
            for pair in handovered_pairs:
                mat[pair[0]][pair[1]]['weight'] += pair[2] * 1 / sf

    sn = DiGraph(mat)
    sn.add_nodes_from(el.groupby(const.RESOURCE).groups.keys())
    return sn
