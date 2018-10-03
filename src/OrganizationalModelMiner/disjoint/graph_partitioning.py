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

def connected_components(sn):
    '''
    This method finds connected components in a given graph and derives the
    organizational model.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.

    Returns:
        ogs: list of frozensets
            A list of organizational groups.
    '''

    print('Applying disjoint organizational model mining using' +
            ' edge threshold:')
    # step 1. obtain the connected components in the resulting graph
    from networkx import connected_components, number_connected_components
    print('Found {} connected components in total.'.format(
        number_connected_components(sn)))
    # step 2. derive the organizational groups from the connected components
    ogs = list()
    for comp in connected_components(sn):
        ogs.append(frozenset(comp))
    print('{} organizational groups discovered.'.format(len(ogs)))
    return ogs

