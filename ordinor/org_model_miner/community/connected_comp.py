"""
Detecting communities based on connected components
"""

from operator import itemgetter
from itertools import combinations

from networkx import Graph
from networkx import restricted_view
from networkx import \
    connected_components as cc, \
    number_connected_components as num_cc
from networkx import \
    strongly_connected_components as scc, \
    number_strongly_connected_components as num_scc
from networkx import betweenness_centrality

from ordinor.utils.validation import check_convert_input_log
from ordinor.social_network_miner.joint_activities import distance
from ordinor.social_network_miner.joint_cases import working_together
from ordinor.org_model_miner._helpers import cross_validation_score
import ordinor.exceptions as exc
import ordinor.constants as const

def _mja(profiles, n_groups, metric='euclidean'):
    print('Applying graph/network-based MJA:')
    sn = distance(profiles, metric=metric, convert=True)

    edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
    # search the cut edge using bisection (i.e. binary search)
    lo = 0
    hi = len(edges_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        sub_sn = restricted_view(
            sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:mid]]
        )
        if num_cc(sub_sn) < n_groups:
            lo = mid + 1
        else:
            hi = mid
    sub_sn = restricted_view(
        sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:lo]]
    )
    ogs = list()
    if num_cc(sub_sn) == n_groups:
        for comp in cc(sub_sn):
            ogs.append(frozenset(comp))
    else:
        raise exc.AlgorithmRuntimeError(
            reason='Cannot discover groups exactly as the specified number',
            suggestion='Try specifying other group numbers?'
        )
    return ogs


def mja(profiles, n_groups, metric='euclidean', search_only=False):
    """
    Apply Metrics based on Joint Activities to discover organizational 
    groups [1]_.

    Parameters
    ----------
    profiles : pandas.DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.
    search_only : bool, optional, default False
        A Boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    See Also
    --------
    ordinor.social_network_miner.joint_activities

    References
    ----------
    .. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
       Discovering social networks from event logs. *Computer Supported
       Cooperative Work (CSCW)*, 14(6), 549-593.
       `<https://doi.org/10.1007/s10606-005-9005-9>`_
    """
    if type(n_groups) is int:
        return _mja(profiles, n_groups, metric)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _mja(profiles, n_groups[0], metric)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
            score = cross_validation_score(
                X=profiles, miner=_mja,
                miner_params={
                    'n_groups': k,
                    'metric': metric
                },
                proximity_metric=metric
            )
            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k 
        else:
            return _mja(profiles, best_k, metric)
    else:
        raise exc.InvalidParameterError(
            param='n_groups',
            reason='Expected an int or a non-empty list'
        )


def _mjc(el, n_groups, method='threshold'):
    print('Applying graph/network-based MJC:')
    # directed graph
    sn = working_together(el, normalize=const.RESOURCE)

    if method == 'threshold':
        # Eliminate less-important edges and maintain the stronger ones
        # Casting from DiGraph to Graph needs to be configured manually
        # otherwise NetworkX would approach this in an arbitrary fashion
        undirected_edge_list = list()
        for pair in combinations(sn.nodes, r=2):
            if (sn.has_edge(pair[0], pair[1]) and 
                sn.has_edge(pair[1], pair[0])):
                undirected_edge_wt = 0.5 * (
                    sn[pair[0]][pair[1]]['weight'] +
                    sn[pair[1]][pair[0]]['weight'])
                undirected_edge_list.append(
                    (pair[0], pair[1], {'weight': undirected_edge_wt}))
            else:
                pass
        sn.clear()
        del sn
        sn = Graph()
        sn.add_edges_from(undirected_edge_list)
        del undirected_edge_list[:]

        exc.warn_data_type_casted('DiGraph', 'Graph')

        edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
        # search the cut edge using bisection (i.e. binary search)
        lo = 0
        hi = len(edges_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            sub_sn = restricted_view(
                sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:mid]]
            )
            if num_scc(sub_sn) < n_groups:
                lo = mid + 1
            else:
                hi = mid
        sub_sn = restricted_view(
            sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:lo]]
        )
    elif method == 'centrality':
        # Disconnect particular nodes
        # betweenness centrality can be calculated on directed graphs
        node_centrality = betweenness_centrality(sn, weight='weight')
        # sorted the nodes by centrality in a descending order
        nodes_sorted = list(map(itemgetter(0), sorted(
            list(node_centrality.items()), key=itemgetter(1), reverse=True))
        )
        # search the disconnected node using bisection (i.e. binary search)
        lo = 0
        hi = len(nodes_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            # discard the connecting links but not the nodes themselves
            nodes_to_disconnect = nodes_sorted[:mid]
            edges_to_disconnect = (
                list(sn.in_edges(nbunch=nodes_to_disconnect)) +
                list(sn.out_edges(nbunch=nodes_to_disconnect)))
            sub_sn = restricted_view(
                sn, nodes=[],
                edges=edges_to_disconnect)
            if num_scc(sub_sn) < n_groups:
                lo = mid + 1
            else:
                hi = mid
        nodes_to_disconnect = nodes_sorted[:lo]
        edges_to_disconnect = (
            list(sn.in_edges(nbunch=nodes_to_disconnect)) +
            list(sn.out_edges(nbunch=nodes_to_disconnect))
        )
        sub_sn = restricted_view(
            sn, nodes=[], edges=edges_to_disconnect
        )

    else:
        raise exc.InvalidParameterError(
            param='method',
            reason='Can only be one of {"threshold", "centrality"}'
        )

    ogs = list()
    if num_scc(sub_sn) == n_groups:
        for comp in scc(sub_sn):
            ogs.append(frozenset(comp))
    return ogs


def mjc(el, n_groups, search_only=False):
    """
    Apply Metrics based on Joint Cases to discover organizational 
    groups [1]_.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    search_only : bool, optional, default False
        A Boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    See Also
    --------
    ordinor.social_network_miner.joint_cases

    References
    ----------
    .. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
       Discovering social networks from event logs. *Computer Supported
       Cooperative Work (CSCW)*, 14(6), 549-593.
       `<https://doi.org/10.1007/s10606-005-9005-9>`_
    """
    el = check_convert_input_log(el)
    if type(n_groups) is int:
        return _mjc(el, n_groups)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _mjc(el, n_groups[0])
    elif type(n_groups) is list and len(n_groups) > 1:
        # TODO: How to evaluate a result from applying MJC?
        raise NotImplementedError
    else:
        raise exc.InvalidParameterError(
            param='n_groups',
            reason='Expected an int or a non-empty list'
        )
