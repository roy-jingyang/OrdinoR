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
        metric='euclidean', use_log_scale=False):
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
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        ogs: list of frozensets
            A list of organizational groups.
    '''
    print('Applying MJA:')
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)
    from operator import itemgetter
    edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
    from networkx import (
            restricted_view, connected_components, number_connected_components)
    for i in range(len(edges_sorted)):
        sub_sn = restricted_view(
                sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:i]])
        if number_connected_components(sub_sn) == n_groups:
            ogs = list()
            for comp in connected_components(sub_sn):
                ogs.append(frozenset(comp))
            print('{} organizational groups discovered.'.format(len(ogs)))
            return ogs
        else:
            pass

    return None

# TODO
def mja(
        profiles, n_groups,
        metric='euclidean', use_log_scale=False):
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int, or iterator
            The (range of) number of groups to be discovered.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        best_ogs: list of frozensets
            A list of organizational groups.
    '''
    if type(n_groups) is int:
        return _mja(profiles, n_groups, metric, use_log_scale)
    else:
        best_ogs = None
        best_score = float('-inf')
        for k in n_groups:
            #TODO: calculate the scores
            if score > best_score:
                best_score = score
                best_ogs = cand_ogs
        print('Selected "K" = {}'.format(len(best_ogs)))
        return best_ogs

def _mjc(
        el, n_groups):
    '''
    This method implements the algorithm of discovering an organizational model
    using edge thresholding in a network built by metrics based on joint
    cases (MJC).

    Params:
        el: DataFrame
            The imported event log.
        n_groups: int
            The number of groups to be discovered.
    Returns:
        ogs: list of frozensets
            A list of organizational groups.
    '''
    print('Applying MJC:')
    from SocialNetworkMiner.joint_cases import working_together
    sn = working_together(el)
    print('[Warning] DiGraph casted to Graph.')
    sn = sn.to_undirected()
    from operator import itemgetter
    edges_sorted = sorted(sn.edges.data('weight'), key=itemgetter(2))
    from networkx import (
            restricted_view, connected_components, number_connected_components)
    for i in range(len(edges_sorted)):
        sub_sn = restricted_view(
                sn, nodes=[], edges=[(u, v) for u, v, w in edges_sorted[:i]])
        if number_connected_components(sub_sn) == n_groups:
            ogs = list()
            for comp in connected_components(sub_sn):
                ogs.append(frozenset(comp))
            print('{} organizational groups discovered.'.format(len(ogs)))
            return ogs
        else:
            pass

    return None

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
        n_groups: int or iterator
            The number of groups to be discovered.
    Returns:
        best_ogs: list of frozensets
            A list of organizational groups.
    '''
    if type(n_groups) is int:
        return _mjc(el, n_groups)
    else:
        pass

