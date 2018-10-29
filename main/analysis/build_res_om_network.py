#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Build resource social network depending on group membership. 

import csv
import sys
import networkx as nx
from collections import defaultdict

f_org_model = sys.argv[1]
f_out_network = sys.argv[2]

if __name__ == '__main__':
    # read organizational model
    org_model = defaultdict(lambda: set())
    with open(f_org_model, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for rid in row[2].split(';'):
                    org_model[row[0]].add(rid)

    G = nx.Graph()
    flag = set()

    # add labels for each resource (node)
    for k, x in org_model.items():
        for r in x:
            if r in flag:
                G.nodes[r]['membership'] = 'MULTI'
            else:
                flag.add(r)
                G.add_node(r, label=r, membership=k) # Gephi requires label
    
    print(len(G))
    # add connecting edge between each two resource of same entities (edge)
    for k, x in org_model.items(): 
        x = list(x)
        for i in range(len(x) - 1):
            for j in range(i + 1, len(x)):
                G.add_edge(x[i], x[j])
    
    nx.write_graphml(G, f_out_network)

