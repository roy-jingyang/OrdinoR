# -*- coding: utf-8 -*-

'''
This module contains the implementation of mining a social network from an
event log, using metrics based on joint activities (ref. van der Aalst et.
al, CSCW 2005).
'''

from networkx import is_directed

def performer_activity_matrix(el, use_log_scale):
    '''
    This method builds a "profile" based on how frequent individuals originate
    events with specific activity names, i.e. the performer-by-activity matrix.

    Params:
        el: DataFrame
            The impoted event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        DataFrame
            The contructed resource profiles as a pandas DataFrame.
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

def distance(profiles, 
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
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of different distance-related metrics.
        convert: boolean, optional
            Boolean flag to determine whether the weight values of the edges in
            the mined network should be converted to similarity flavored
            measure. The default is to keep the original distance values 
            ranged [0, inf).
    Returns:
        sn: NetworkX Graph
            The mined social network as a NetworkX Graph object.
    '''
    from scipy.spatial.distance import squareform, pdist
    x = squareform(pdist(profiles, metric=metric)) # preserve index

    if convert:
        print('[Warning] Distance measure has been converted to similarity.')
        # NOTE: different strategies may be employed for the transformation
        # from distance measure to similarity measure
        min_v = x.min()
        max_v = x.max()
        x = 1 - (x - min_v) / (max_v - min_v)
    # convert to Graph
    from networkx import Graph, relabel_nodes
    from numpy import fill_diagonal
    fill_diagonal(x, 0) # ignore self-loops
    G = Graph(x)
    # relabel nodes using resource index in the profile matrix
    nodes = list(G.nodes)
    node_mapping = dict()
    for i in range(len(x)):
        node_mapping[nodes[i]] = profiles.index[i] 
    sn = relabel_nodes(G, node_mapping)
    sn.add_nodes_from(profiles.index)
    if not sn.is_directed():
        return sn
    else:
        exit('[Error] Social network based on joint activities found directed')

