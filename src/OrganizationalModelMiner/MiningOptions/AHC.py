#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networx as nx
from collections import defaultdict

def cluster(graph, k_clusters):
    print('Applying Agglomerative Hierarchical Clustering:')
    k = len(graph.nodes)
    if k_clusters > k:
        print('Error: Desired #clusters > #nodes.')
        exit(1)
    else:
        clusters = list()
        for u in graph.nodes:
            s = set()
            s.add(u)
            clusters.append(s)
        while k > k_clusters:
            k -= 1
            # obtain the pair of clusters with minimum cluster distance
            cluster_dist = list()
            for i in range(len(clusters) - 1):
                ci = clusters[i]
                for j in range(i, len(clusters)):
                    cj = clusters[j]
                    # define cluster distance as min. distance between member nodes
                    node_dist = list()
                    for u in ci:
                        for v in cj:
                            node_dist.append(graph[u][v])
                    cluster_dist.append((i, j, min(node_dist)))
            min_cluster_dist = min(cluster_dist, key=lambda x: x[2])
            # merge closest clusters
            merged = clusters[min_cluster_dist[0]].union(
                    clusters[min_cluster_dist[1])
            clusters.append(merged)
            del clusters[min_cluster_dist[0]]
            del clusters[min_cluster_dist[1]]

        entities = dict()
        entity_id = -1
        for c in clusters:
            entity_id += 1
            entities[entity_id] = c
        print('{} organizational entities extracted.'.format(len(entities))
        return copy.deepcopy(entities)

