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

def remove_edges(sn, threshold, weight):
    '''
    This method removes edges from a given graph with weight less than a speci
    -fied threshold.

    Params:
        sn:
        threshold:
        weight:
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''
    print('Removing edges from graph with weighting values < {:.f}'.format(
        threshold))
    pass

