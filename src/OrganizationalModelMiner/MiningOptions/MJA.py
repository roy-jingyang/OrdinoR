#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networx as nx
from collections import defaultdict

def threshold(graph, threshold_value):
    print('Applying Metrics based on Joint Activities:')
    num_edges_old = len(graph.edges)
    # iterate through all edges
    edges_to_remove = list()
    for u, v, wt in graph.edges.data('weight'):
        # remove edge if weight lower than threshold
        if wt < threshold_value:
            edges_to_remove.append((u, v))
    graph.remove_edges_from(edges_to_remove)
    print('{:.2%}% edges have been filtered by threshold
            {}.'.format(len(edges_to_remove) / num_edges_old), threshold_value) 
    # obtain the connected components as discovered results
    entities = defaultdict(lambda: set())
    entity_id = -1 # consecutive numbers as entity id
    for comp in nx.connected_components(graph):
        entity_id += 1
        for u in list(comp):
            entities[entity_id].add(u)
    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

