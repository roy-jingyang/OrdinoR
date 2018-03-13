#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist
from collections import defaultdict
from EvaluationOptions import Unsupervised

#def threshold(cases, threshold_value_step):
def mine(f_sn_model):
    print('Applying Community Detection (Appice):')

    # read gml format
    G = nx.read_gml(f_sn_model)

    # 1. Build the linear network using the original network
    lg = nx.DiGraph()
    for (u, v, w) in G.edges.data('weight'):
        lg.add_node('{}><{}'.format(str(u), str(v)))

    edges = list(G.edges.data('weight'))
    for i in range(len(edges) - 1):
        ei = edges[i]
        for j in range(i + 1, len(edges)):
            ej = edges[j]
            x = None
            if ei[0] == ej[0] or ei[0] == ej[1]:
                x = ei[0]
            elif ei[1] == ej[0] or ei[1] == ej[1]:
                x = ei[1]
            if x is not None:
                # i -> j
                w_l = ei[2] / (G.degree(nbunch=x, weight='weight') - ej[2])
                # i <- j
                w_r = ej[2] / (G.degree(nbunch=x, weight='weight') - ei[2])
                lg.add_edge(
                        '{}><{}'.format(str(ei[0]), str(ei[1])),
                        '{}><{}'.format(str(ej[0]), str(ej[1])),
                        weight=w_l)
                lg.add_edge(
                        '{}><{}'.format(str(ej[0]), str(ej[1])),
                        '{}><{}'.format(str(ei[0]), str(ei[1])),
                        weight=w_r)
                            
    nx.write_gml(lg, 'linear_network.tmp.gml')
    del lg
    print('Transformed linear network exported.')
    print('Run external software to discover communities.')

    # 2. Run Louvain algorithm on the linear network to discover communities
    # Run using external software, then load the results back in here
    # Gephi "Statistics" -> "Modularity", result exported as *.gml
    # Pajek "Network" -> "Create Partitions" -> "Communities"
    print('Input file: ', end='')
    f_grouped_lg = input()
    grouped_lg = nx.read_gml(f_grouped_lg)
    print('Linear network with detected communities imported.')

    # 3. Map communities onto the original network to get overlapping results
    # find the one-to-one mapping
    entities = defaultdict(lambda: set())

    linear_communities = defaultdict(lambda: list())
    for ln, lnattr in grouped_lg.nodes(data=True):
        linear_communities[lnattr['ModularityClass']].append(ln)
    for lc_label, lns in linear_communities.items():
        for ln in lns:
            u = ln.split('><')[0]
            v = ln.split('><')[1]
            entities[lc_label].add(u)
            entities[lc_label].add(v)

    # Optional: calculate the degree of membership

    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

