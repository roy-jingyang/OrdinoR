#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
from collections import defaultdict

def mine(f_sn_model):
    print('Applying Community Detection (Appice):')

    # read gml format
    G = nx.read_gml(f_sn_model)
    edges = sorted(list(G.edges.data('weight')))
    #edges_dict = nx.to_dict_of_dicts(G)

    # 1. Build the linear network using the original network
    with open('linear_network.tmp.net', 'w') as f_pajek:
        f_pajek.write('*Vertices {}\n'.format(len(edges)))

        # create nodes in linear network
        for i in range(len(edges)):
            e = edges[i]
            u = e[0]
            v = e[1]
            f_pajek.write('{} "{}><{}"\n'.format(
                i + 1, str(u), str(v)))

        print('{} nodes in the linear network'.format(len(edges)))

        cnt = 0
        # create edges in linear network
        f_pajek.write('*arcs\n')
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
                    
                    f_pajek.write('{} {} {:.9f}\n'.format(
                        i + 1, j + 1, w_l))
                    f_pajek.write('{} {} {:.9f}\n'.format(
                        j + 1, i + 1, w_r))

                    cnt += 2
                    if cnt % 100000 == 0:
                        print('{} edges in the linear network created'.format(
                            cnt))
        '''
        for e in edges:
            u = e[0]
            v = e[1]
            w = e[2]
            
            # all other edges that share node u
            for node, weight in edges_dict[u].items():
                deg = G.degree(nbunch=u, weight='weight')
                wu_l = w / (deg - weight)
                wu_r = weight / (deg - w)

            # all other edges that share node v
            for node, weight in edges_dict[v].items():
                deg = G.degree(nbunch=v, weight='weight')
                wv_l = w / (deg - weight)
                wv_r = weight / (deg - w)
        '''

        print('Transformed linear network exported.')
        print('Run external software to discover communities.')

    # 2. Run Louvain algorithm on the linear network to discover communities
    # Run using external software, then load the results back in here
    # Gephi "Statistics" -> "Modularity", result exported as *.gml
    # Pajek "Network" -> "Create Partitions" -> "Communities"
    print('Input file: ', end='')
    f_pajek_clu = input()
    print('Linear network with detected communities imported.')

    # 3. Map communities onto the original network to get overlapping results
    # find the one-to-one mapping
    entities = defaultdict(lambda: set())

    cnt = 0
    is_header_line = True
    with open(f_pajek_clu, 'r') as f:
        for line in f:
            if is_header_line:
                is_header_line = False
            else:
                label = line.strip()
                u = edges[cnt][0]
                v = edges[cnt][1]
                entities[label].add(u)
                entities[label].add(v)
                cnt += 1

    # Optional: calculate the degree of membership

    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

