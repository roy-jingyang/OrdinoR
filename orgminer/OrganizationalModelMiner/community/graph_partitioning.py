# -*- coding: utf-8 -*-

"""This module contains the implementation of graph/network -based 
organizational mining methods, based on the use of graph partitioning 
techniques.
"""
from warnings import warn

def _mja(profiles, n_groups, metric='euclidean'):
    """Apply Metrics based on Joint Activities to discover organizational 
    groups [1]_.

    Parameters
    ----------
    profiles : DataFrame Constructed resource profiles. n_groups : int
        Expected number of resource groups. 
    metric : str, optional, default 'euclidean' 
        Choice of metrics for measuring the distance while calculating
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    ogs : list of frozensets Discovered resource groups.

    Raises
    ------
    RuntimeError If the specified number of groups could not be
        discovered.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities

    References
    ----------
    .. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
       Discovering social networks from event logs. *Computer Supported
       Cooperative Work (CSCW)*, 14(6), 549-593.
       `<https://doi.org/10.1007/s10606-005-9005-9>`_
    """
    print('Applying graph/network-based MJA:')
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    from operator import itemgetter
    edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
    from networkx import restricted_view
    from networkx import connected_components, number_connected_components
    # search the cut edge using bisection (i.e. binary search)
    lo = 0
    hi = len(edges_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        sub_sn = restricted_view(
            sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:mid]])
        if number_connected_components(sub_sn) < n_groups:
            lo = mid + 1
        else:
            hi = mid
    sub_sn = restricted_view(
        sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:lo]])
    ogs = list()
    if number_connected_components(sub_sn) == n_groups:
        for comp in connected_components(sub_sn):
            ogs.append(frozenset(comp))
    else:
        raise RuntimeError('Unable to discover specified number of groups.')
    return ogs


def mja(profiles, n_groups, metric='euclidean', search_only=False):
    """Apply Metrics based on Joint Activities to discover organizational 
    groups [1]_.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
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

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities

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
        from orgminer.OrganizationalModelMiner.utilities import \
            cross_validation_score
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
        raise TypeError('Invalid type for parameter `{}`: {}'.format(
            'n_groups', type(n_groups)))


def _mjc(el, n_groups, method='threshold'):
    """Apply Metrics based on Joint Cases to discover organizational 
    groups [1]_.

    Parameters
    ----------
    el : DataFrame
        An event log.
    n_groups : int
        Expected number of resource groups.
    method : {'threshold', 'centrality'}, optional, default 'threshold'
        Options for the method to be used for finding graph components. 
        Could be one of the following:

            - 'threshold': using edge thresholding on edges to remove
              links.
            - 'centrality': disconnect nodes with high betweenness 
              centrality, i.e., shortest-path centrality.

    Returns
    -------
    ogs : list of frozensets
        Discovered resource groups.
    
    Raises
    ------
    ValueError
        If the specified method for finding graph components is invalid.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_cases

    References
    ----------
    .. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
       Discovering social networks from event logs. *Computer Supported
       Cooperative Work (CSCW)*, 14(6), 549-593.
       `<https://doi.org/10.1007/s10606-005-9005-9>`_
    """
    print('Applying graph/network-based MJC:')
    from orgminer.SocialNetworkMiner.joint_cases import working_together
    # directed graph
    sn = working_together(el, normalize='resource')

    from networkx import restricted_view
    from networkx import strongly_connected_components as cc
    from networkx import number_strongly_connected_components as num_cc

    if method == 'threshold':
        # Eliminate less-important edges and maintain the stronger ones
        # Casting from DiGraph to Graph needs to be configured manually
        # otherwise NetworkX would approach this in an arbitrary fashion
        from itertools import combinations
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
        from networkx import Graph
        sn = Graph()
        sn.add_edges_from(undirected_edge_list)
        del undirected_edge_list[:]
        warn('DiGraph casted to Graph.', RuntimeWarning)

        from operator import itemgetter
        edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
        # search the cut edge using bisection (i.e. binary search)
        lo = 0
        hi = len(edges_sorted)
        while lo < hi:
            mid = (lo + hi) // 2
            sub_sn = restricted_view(
                sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:mid]])
            if num_cc(sub_sn) < n_groups:
                lo = mid + 1
            else:
                hi = mid
        sub_sn = restricted_view(
            sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:lo]])
    elif method == 'centrality':
        # Disconnect particular nodes
        from networkx import betweenness_centrality
        # betweenness centrality can be calculated on directed graphs
        node_centrality = betweenness_centrality(sn, weight='weight')
        from operator import itemgetter
        # sorted the nodes by centrality in a descending order
        nodes_sorted = list(map(itemgetter(0), sorted(
            list(node_centrality.items()), key=itemgetter(1), reverse=True)))
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
            if num_cc(sub_sn) < n_groups:
                lo = mid + 1
            else:
                hi = mid
        nodes_to_disconnect = nodes_sorted[:lo]
        edges_to_disconnect = (
            list(sn.in_edges(nbunch=nodes_to_disconnect)) +
            list(sn.out_edges(nbunch=nodes_to_disconnect)))
        sub_sn = restricted_view(
            sn, nodes=[], edges=edges_to_disconnect)

    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'method', method))

    ogs = list()
    if num_cc(sub_sn) == n_groups:
        for comp in cc(sub_sn):
            ogs.append(frozenset(comp))
    else:
        pass
    return ogs


def mjc(el, n_groups, search_only=False):
    """Apply Metrics based on Joint Cases to discover organizational 
    groups [1]_.

    Parameters
    ----------
    el : DataFrame
        An event log.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
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

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_cases

    References
    ----------
    .. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
       Discovering social networks from event logs. *Computer Supported
       Cooperative Work (CSCW)*, 14(6), 549-593.
       `<https://doi.org/10.1007/s10606-005-9005-9>`_
    """
    if type(n_groups) is int:
        return _mjc(el, n_groups)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _mjc(el, n_groups[0])
    elif type(n_groups) is list and len(n_groups) > 1:
        # TODO: How to evaluate a result from applying MJC?
        raise NotImplementedError
    else:
        raise TypeError('Invalid type for parameter `{}`: {}'.format(
            'n_groups', type(n_groups)))

