# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining disjoint organiza-
tional models, based on the use of graph partitioning by egde removal. A graph
(NetworkX graph, weighted) is expected to be used as input. Methods include:
    1. Thresholding edges in a graph built by MJA (Song & van der Aalst)
    2. Thresholding edges in a graph built by MJC (Song & van der Aalst)
'''

def _mja(
        profiles, n_groups,
        metric='euclidean'):
    '''
    This method implements the algorithm of discovering an organizational model
    using edge thresholding in a network built by metrics based on joint
    activities (MJA).

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
    Returns:
        ogs: list of frozensets
            A list of organizational groups.
    '''
    print('Applying MJA:')
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    from operator import itemgetter
    edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
    from networkx import (
        restricted_view, connected_components, number_connected_components)
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
        pass
    #print('{} organizational groups discovered.'.format(len(ogs)))
    return ogs

def mja(
        profiles, n_groups,
        metric='euclidean',
        search_only=False):
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: iterable
            The (range of) number of groups to be discovered.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        search_only: boolean, optional
            Determine whether to search for the number of groups only or to
            perform cluster analysis based on the search result. The default is
            to perform cluster analysis after searching.
    Returns:
        best_ogs: list of frozensets
            A list of organizational groups.
    '''
    if len(n_groups) == 1:
        return _mja(profiles, n_groups[0], metric)
    else:
        from orgminer.OrganizationalModelMiner.utilities import cross_validation_score
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

def _mjc(
        el, n_groups, method='threshold'):
    '''
    This method implements the algorithm of discovering an organizational model
    using a network built by metrics based on joint cases (MJC).

    Params:
        el: DataFrame
            The imported event log.
        n_groups: int
            The number of groups to be discovered.
        method: str
            The option designating the method to be used for finding
            sub-networks, can be either of:
                - 'threshold': using edge thresholding on edges to remove
                  links. Default.
                - 'centrality': disconnect nodes with high betweenness 
                  centrality, i.e. shortest-path centrality.
    Returns:
        ogs: list of frozensets
            A list of organizational groups.
        sn: NetworkX (Di)Graph
            A resource social network used for discovering groups.
    '''
    print('Applying MJC:')
    from orgminer.SocialNetworkMiner.joint_cases import working_together
    sn = working_together(el, normalize='resource')

    from networkx import restricted_view
    if sn.is_directed:
        from networkx import strongly_connected_components as cc
        from networkx import number_strongly_connected_components as num_cc
    else:
        from networkx import connected_components as cc
        from networkx import number_connected_components as num_cc

    if method == 'threshold':
        # Eliminate less-important edges and maintain the stronger ones

        '''
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
        print('[Warning] DiGraph casted to Graph.')
        '''

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
        exit('[Error] Unrecognized option for methods')

    ogs = list()
    if num_cc(sub_sn) == n_groups:
        for comp in cc(sub_sn):
            ogs.append(frozenset(comp))
    else:
        pass
    #print('{} organizational groups discovered.'.format(len(ogs)))
    return ogs, working_together(el, normalize='resource')

# TODO: How to evaluate a result from applying MJC?
def mjc(
        el, n_groups):
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        el: DataFrame
            The imported event log.
        n_groups: iterable
            The (range of) number of groups to be discovered.
    Returns:
        best_ogs: list of frozensets
            A list of organizational groups.
    '''
    if type(n_groups) is int:
        return _mjc(el, n_groups)
    else:
        pass

