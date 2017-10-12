#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
from collections import defaultdict

# TODO: need to re-name according to the clustering strategy
def cluster(graph, k_clusters):
    print('Applying Agglomerative Hierarchical Clustering:')
    k = len(graph.nodes)
    if k_clusters > k:
        print('Error: Desired #clusters > #nodes.')
        exit(1)
    else:
        clusters = list()
        # initialize #cluster = #resources
        for u in graph.nodes:
            s = set()
            s.add(u)
            clusters.append(s)
        # decrease #cluster iteratively, by:
        while k > k_clusters:
            k -= 1
            # obtain the pair of clusters with minimum cluster distance
            cluster_dist = list()
            for i in range(len(clusters) - 1):
                ci = clusters[i]
                for j in range(i + 1, len(clusters)):
                    cj = clusters[j]
                    # define cluster distance as min. distance between member nodes
                    node_dist = list()
                    for u in ci:
                        for v in cj:
                            if u == v:
                                node_dist.append(0)
                            else:
                                node_dist.append(graph[u][v]['weight'])
                    cluster_dist.append((i, j, min(node_dist)))
            min_cluster_dist = min(cluster_dist, key=lambda x: x[2])
            # merge closest clusters
            min_cluster_x = clusters[min_cluster_dist[0]]
            min_cluster_y = clusters[min_cluster_dist[1]]
            #print('k={}, Closest:'.format(k), end='\t')
            #print(min_cluster_x, end='\t')
            #print(min_cluster_y)
            clusters.remove(min_cluster_x)
            clusters.remove(min_cluster_y)
            merged = min_cluster_x.union(min_cluster_y)
            #print(merged)
            clusters.append(merged)

        entities = defaultdict(lambda: set())
        entity_id = -1
        for c in clusters:
            entity_id += 1
            for n in c:
                entities[entity_id].add(n)
        print('{} organizational entities extracted.'.format(len(entities)))
        return copy.deepcopy(entities)

