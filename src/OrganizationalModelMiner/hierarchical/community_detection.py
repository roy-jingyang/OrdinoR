# -*- coding: utf-8 -*-

'''
This module contains the implementation of hierarchical organizational mining
methods, based on the use of community detection techniques. Methods include:
    1. The Girvan-Newman algorithm
'''

def betweenness(
        sn, n_groups,
        weight=None):
    '''
    This method implements the traditional Girvan-Newman algorithm designed for
    community detection, in which the edge with the highest betweenness is re-
    moved at each iteration and the network breaks down gradually into frac-
    tions. During the procedure, a hierarchical structure of the communities
    is exposed.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.
        n_groups: int
            The number of groups to be discovered.
        weight: str, optional
            String specifiying the name of the weight attribute in the NetworkX
            graph to be recognized as weight values. The default is None, i.e.
            the graph is considered as unweighted.
    Returns:
        og: dict of frozensets
            The mined organizational groups.
        og_hcy: DataFrame
            The hierarchical structure as a pandas DataFrame, with resource ids
            as indices, levels of the hierarchy as columns, and group ids as
            the values, e.g. for 20 resources placed in a 5-level hierarhical
            structure with 8 groups at the lowest level, there should be 20
            rows and 5 columns in the DataFrame, and the values should be in
            range of 0 to 7.
    '''
    
    # TODO
    from numpy import zeros
    # this algorithm removes one edge at each step
    resource_idx = sorted(sn.nodes)
    mx_tree = zeros(shape=(len(resource_idx), n_groups), dtype=int)
    from networkx.algorithms.community.centrality import girvan_newman
    # TODO
    if weight is not None:
        from networkx import edge_betweenness_centrality
        def most_central_edge(G):
            centrality = edge_betweenness_centrality(G, weight=weight)
            return max(centrality, key=centrality.get)
        comp = girvan_newman(sn, most_valuable_edge=most_central_edge)
    else:
        comp = girvan_newman(sn)
    level = 0 # the first column (level) of mx_tree is all 0s
    # obtain communities at each level until the specific number of group found
    from itertools import takewhile
    for communities in takewhile(
            lambda communities: len(communities) <= n_groups, comp):
        level += 1
        community_idx = 0 # count from 0
        for c in communities:
            for r in sorted(c): # update the mark for each resource
                mx_tree[resource_idx.index(r)][level] = community_idx
            community_idx += 1 # move to the next community
    # convert to a matrix and wrap as DataFrame og_hcy
    from pandas import DataFrame
    og_hcy = DataFrame(mx_tree, index=resource_idx) # preserve indices

    print(og_hcy)

    from collections import defaultdict
    groups = defaultdict(lambda: set())
    # add by each resource
    for i in range(len(og_hcy.index)):
        groups[og_hcy.iloc[i,-1]].add(og_hcy.index[i])

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()], og_hcy

