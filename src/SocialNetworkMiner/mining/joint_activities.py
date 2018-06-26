#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on joint activities (ref. van der Aalst et.
al, CSCW 2005).
'''

def build_performer_activity_matrix(c, use_log_scale):
    '''
    This method builds a "profile" based on how frequent individuals conduct
    specific activities. The "performer by activity matrix" is used to repre-
    sent these profiles.

    Params:
        c: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        pam: DataFrame
            The constructed performer by activity matrix as a pandas DataFrame,
            with resource ids as indices and activity names as columns.
    '''

    from collections import defaultdict
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for case_id, trace in c.groupby('case_id'):
        for event in trace.itertuples():
            pam[event.activity][event.resource] += 1

    from pandas import DataFrame
    if use_log_scale: 
        return DataFrame(pam)
    else:
        from numpy import log
        return DataFrame(pam).apply(lambda x: log(x + 1))

def distance(c,
        use_log_scale=False,
        metric='euclidean'):
    '''
    This method implements the mining based on metrics based on joint activi-
    ties where distance-related measures are used. Notice that the weight
    values correspond to the distances between the "profiles", thus:
        1. A higher weight value means farther relationships, which is
        different with other metrics, and
        2. The generated network is naturally a undirected graph.

    Params:
        c: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
        metric: str, optional
            Choice of different distance-related metrices. Options include:
                - 'cityblock': the Manhattan (Rectilinear) distance
                - 'euclidean': the Euclidean distance, default
                - 'hamming': the Hamming distance (the default threshold is 0).
    Returns:
        sn: NetworkX Graph
            The mined social network as a Network Graph object.
    '''

    pam = build_performer_activity_matrix(c, use_log_scale)
    from scipy.spatial.distance import squareform, pdist
    x = squareform(pdist(pam, metric=metric)) # preserve index
    # convert to Graph
    from networkx import Graph, relabel_nodes
    from numpy import fill_diagonal
    fill_diagonal(x, 0) # ignore self-loops
    G = Graph(x)
    # relabel nodes using resource index in the profile matrix
    nodes = list(G.nodes)
    node_mapping = dict()
    for i in range(len(x)):
        node_mapping[nodes[i]] = pam.index[i] 
    sn = relabel_nodes(G, node_mapping)
    sn.add_nodes_from(c.groupby('resource').groups.keys())
    return sn

def correlation(c,
        use_log_scale=False,
        metric='pearson'):
    '''
    This method implements the mining based on metrics based on joint activi-
    ties where correlation-related measures are used.

    Params:
        c: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
        metric: str, optional
            Choice of different distance-related metrices. Options include:
                - 'pearson': the Pearson Correlation Coefficient (PCC), default
    Returns:
        sn: NetworkX Graph
            The mined social network as a Network Graph object.
    '''
    
    pam = build_performer_activity_matrix(c, use_log_scale)
    from scipy.spatial.distance import squareform, pdist
    if metric == 'pearson':
        x = squareform(pdist(pam, metric='correlation')) # preserve index
    else:
        pass
    # convert to Graph
    from networkx import Graph, relabel_nodes
    from numpy import fill_diagonal
    x = 1 - x # correlation rather than 'correlation distance'
    fill_diagonal(x, 0) # ignore self-loops
    G = Graph(x)
    # relabel nodes using resource index in the profile matrix
    nodes = list(G.nodes)
    node_mapping = dict()
    for i in range(len(x)):
        node_mapping[nodes[i]] = pam.index[i] 
    sn = relabel_nodes(G, node_mapping)
    sn.add_nodes_from(c.groupby('resource').groups.keys())
    return sn

