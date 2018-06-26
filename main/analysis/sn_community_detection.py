#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_network = sys.argv[1]
fnout_communities = sys.argv[2]
fnout_commu_hierarchy = sys.argv[3]

if __name__ == '__main__':
    import networkx as nx

    # read network as input (GraphML format)
    sn = nx.read_graphml(fn_network)

    # perform community detection on the imported network
    from networkx.algorithms import community
    commu_generator = community.girvan_newman(sn)
    commu = next(commu_generator)
    while commu:
        print(commu)
        commu = next(commu_generator)

    # export the result to files

