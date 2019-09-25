#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Build resource social network depending on group membership. 

import sys
import networkx as nx
from collections import defaultdict
from itertools import combinations

from OrganizationalModelMiner.base import OrganizationalModel

fn_org_model = sys.argv[1]
fnout_network = sys.argv[2]

if __name__ == '__main__':
    # read organizational model
    with open(fn_org_model, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)

    G = nx.Graph()
    groups = om.find_all_groups()

    # add nodes
    for g in groups:
        for r in g:
            if r in G:
                pass
            else:
                resource_labels = om.find_group_ids(r)
                if len(resources_labels) == 1:
                    membership = str(resource_labels[0])
                    mtype = 'single'
                else:
                    membership = ','.join(resource_labels)
                    mtype = 'multi' 
                G.add_node(r, label=str(r), membership=membership, mtype=mtype) 

    # add edges
    for k, g in enumerate(groups):
        for ur, vr in combinations(g, r=2):
            G.add_edge(ur, vr, group=k)

    nx.write_graphml(G, fnout_network)

