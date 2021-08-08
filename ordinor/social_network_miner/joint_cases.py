"""
Mining social networks from an event log, using metrics based on joint
cases [1]_

See Also
--------
ordinor.social_network_miner.causality
ordinor.social_network_miner.joint_activities

References
----------
.. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
   Discovering social networks from event logs. *Computer Supported
   Cooperative Work (CSCW)*, 14(6), 549-593.
   `<https://doi.org/10.1007/s10606-005-9005-9>`_
"""

from collections import defaultdict
from itertools import permutations

from networkx import DiGraph, Graph

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

def working_together(el, normalize=None):
    """
    Discover a social network from an event log based on working 
    together metric, considering how often two resources were involved 
    in the same case.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
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
    """
    el = check_convert_input_log(el)
    mat = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    for case_id, events in el.groupby(const.CASE_ID):
        participants = set(events[const.RESOURCE])
        # for each pair of participants simultaneously appeared
        for pair in permutations(participants, r=2):
            mat[pair[0]][pair[1]]['weight'] += 1

    if normalize is None:
        is_directed_sn = False
    elif normalize == 'resource':
        is_directed_sn = True
        # count for number of cases a resource participated
        res_case_count = defaultdict(lambda: 0)
        for res_case, events in el.groupby([const.RESOURCE, const.CASE_ID]):
            res_case_count[res_case[0]] += 1
        for r, counts in mat.items():
            for o in counts.keys():
                counts[o]['weight'] /= res_case_count[r]
    elif normalize == 'total':
        is_directed_sn = False
        total_num_cases = len(set(el[const.CASE_ID]))
        for r, counts in mat.items():
            for o in counts.keys():
                counts[o]['weight'] /= total_num_cases
    else:
        raise exc.InvalidParameterError(
            param='normalize',
            reason='Can only be one of {"resource", "total", None}'
        )        

    if is_directed_sn:
        sn = DiGraph(mat)
    else:
        sn = Graph(mat)

    # include isolates
    sn.add_nodes_from(el.groupby(const.RESOURCE).groups.keys())
    return sn
