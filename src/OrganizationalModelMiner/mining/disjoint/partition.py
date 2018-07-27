# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining disjoint organiza-
tional models, based on the use of graph partitioning by egde removal. A graph
(NetworkX graph, weighted) is expected to be used as input. Methods include:
    1. Thresholding edges in a graph built on MJA (Song & van der Aalst)
    2. Thresholding edges in a graph built on MJC (Song & van der Aalst)

Method 1 and 2 vary only in terms of the input graph, therefore the method
'remove_edges' should be used for implementing both algorithms.
'''

def remove_edges(sn, threshold, weight='weight'):
    '''
    This method removes edges from a given graph with weight less than a speci
    -fied threshold (i.e. less import edges).

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.
        threshold: float
            The user-specified threshold value.
        weight: str, optional
            String specifiying the name of the weight attribute in the NetworkX
            graph to be recognized as weight values. The default is 'weight',
            which is the default weight attribute name defined by NetworkX.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''
    from collections import defaultdict

    print('Applying disjoint organizational model mining using' +
            'edge threshold:')
    # step 1. iterate over all edges and mark those with lower weight values
    n_edges_old = len(sn.edges)
    edges_to_be_removed = list()
    for u, v, wt in sn.edges.data(weight):
        if wt < threshold:
            edges_to_be_removed.append((u, v))
    # TODO: add a function for determining the appropriate threshold
    # step 2. remove the edges in place
    sn.remove_edges_from(edges_to_be_removed)
    print('{}/{} edges ({:.2%}) have been removed'.format(
        len(edges_to_be_removed), n_edges_old,
        (len(edges_to_be_removed) / n_edges_old)), end=', ')
    # step 3. obtain the connected components in the resulting graph
    from networkx import connected_components, number_connected_components
    print('resulting in {} connected components in total.'.format(
        number_connected_components(sn)))
    # step 4. derive the organizational model from the connected components
    og = defaultdict(lambda: set())
    group_id = 0
    for comp in connected_components(sn):
        for r in list(comp):
            og[group_id].add(r)
        group_id += 1
    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

