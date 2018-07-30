# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining disjoint organiza-
tional models, based on the use of graph partitioning by egde removal. A graph
(NetworkX graph, weighted) is expected to be used as input. Methods include:
    1. Thresholding edges in a graph built on MJA (Song & van der Aalst)
    2. Thresholding edges in a graph built on MJC (Song & van der Aalst)

Method 1 and 2 vary only in terms of the input graph, therefore the method
'connected_comp' should be used for implementing both algorithms.
'''

def connected_comp(sn):
    '''
    This method finds connected components in a given graph and derives the
    organizational model.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''
    from collections import defaultdict

    print('Applying disjoint organizational model mining using' +
            ' edge threshold:')
    # step 1. obtain the connected components in the resulting graph
    from networkx import connected_components, number_connected_components
    print('Found {} connected components in total.'.format(
        number_connected_components(sn)))
    # step 2. derive the organizational model from the connected components
    og = defaultdict(lambda: set())
    group_id = 0
    for comp in connected_components(sn):
        for r in list(comp):
            og[group_id].add(r)
        group_id += 1
    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

