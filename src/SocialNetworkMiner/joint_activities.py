# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on joint activities (ref. van der Aalst et.
al, CSCW 2005).
'''

def performer_activity_matrix(el, use_log_scale):
    '''
    This method builds a "profile" based on how frequent individuals originate
    events with specific activity names, i.e. the performer-by-activity matrix.

    Params:
        rl: DataFrame
            The resource log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        pam: DataFrame
            The constructed performer by activity matrix as a pandas DataFrame,
            with resource ids as indices and activity names as columns.
        X: DataFrame
            The contructed resource profiles.
    '''

    from collections import defaultdict
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in el.groupby('resource'):
        for event in trace.itertuples():
            pam[res][event.activity] += 1

    from pandas import DataFrame
    if use_log_scale: 
        from numpy import log
        return DataFrame.from_dict(pam, orient='index').fillna(0).apply(
                lambda x: log(x + 1))
    else:
        return DataFrame.from_dict(pam, orient='index').fillna(0)

def distance(el, 
        use_log_scale=False,
        metric='euclidean',
        convert=False):
    '''
    This method implements the mining based on metrics based on joint activi-
    ties where distance-related measures are used. Notice that the weight
    values correspond to the distances between the "profiles", thus:
        1. A HIGHER weight value means FARTHER relationships, which is
        DIFFERENT with other metrics, and
        2. The generated network is naturally a undirected graph.

    Params:
        el: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
        metric: str, optional
            Choice of different distance-related metrices. Options include:
                - 'cityblock': the Manhattan (Rectilinear) distance
                - 'euclidean': the Euclidean distance, default
                - 'hamming': the Hamming distance (the default threshold is 0).
        convert: boolean, optional
            Boolean flag to determine whether the weight values of the edges in
            the mined network should be converted to similarity flavored (i.e.
            first reverse the sign, and scale to range from 0 to 1). The 
            default is to keep the original distance values.
    Returns:
        sn: NetworkX Graph
            The mined social network as a NetworkX Graph object.
    '''
    
    pam = performer_activity_matrix(el, use_log_scale)
    from scipy.spatial.distance import squareform, pdist
    x = squareform(pdist(pam, metric=metric)) # preserve index

    if convert:
        print('[Warning] Distance measure has been converted.')
        x = 0 - x
        x -= x.min()
        x /= x.max() - x.min() 
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
    sn.add_nodes_from(el.groupby('resource').groups.keys())
    return sn

def correlation(el,
        use_log_scale=False,
        metric='pearson'):
    '''
    This method implements the mining based on metrics based on joint activi-
    ties where correlation-related measures are used.

    Params:
        el: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
        metric: str, optional
            Choice of different distance-related metrices. Options include:
                - 'pearson': the Pearson Correlation Coefficient (PCC), default
    Returns:
        sn: NetworkX Graph
            The mined social network as a NetworkX Graph object.
    '''
    
    pam = performer_activity_matrix(el, use_log_scale)
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
    sn.add_nodes_from(el.groupby('resource').groups.keys())
    return sn
