#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import networkx as nx
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3

from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

def _make_profiles(cases):
    # constructing the performer-activity matrix from event logs
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = []

    for caseid, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            if activity in activity_index:
                pass
            else:
                activity_index.append(activity)
            counting_mat[resource][activity] += 1

    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count

    # logrithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    return profile_mat, resource_index

# [DEPRECATED] self-implemented hierarchical clustering algorithm
'''
def cluster(graph, k_cluster):
    print('Applying Agglomerative Hierarchical Clustering:')
    graph = graph.to_undirected()
    k = len(graph.nodes)
    if k_cluster > k:
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
        while k > k_cluster:
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
'''

#TODO
def single_linkage(cases, num_c):
    print('Applying Hierarchical Clustering - single linkage (Nearest Point):')
    profile_mat, resource_index = _make_profiles(cases)
    k = len(profile_mat)
    k_cluster = int(num_c)

    if k_cluster > k:
        print('Error: Desired #clusters > #resources.')
        exit(1)
    else:
        # perform single linkage clustering
        Z = hierarchy.linkage(
                profile_mat, method='single', metric='correlation')

        entities = defaultdict(lambda: set())
        hierarchy_dist_matrix = np.zeros((k, k))
        hierarchy_dist_matrix = defaultdict(lambda: defaultdict(
            lambda: {'weight': 0}))
        # cut the tree to obtain k clusters
        cuttree = hierarchy.cut_tree(Z)
        for i in range(len(cuttree)):
            # indexing EQ to: hierarchy.cut_tree(Z, n_clusters=k_cluster)
            group_id = cuttree[i, k - k_cluster]
            u = resource_index[i]
            entities[group_id].add(resource_index[i])
            # as well as the pairwise distance given the hierarchy
            for j in range(i + 1, len(cuttree)):
                v = resource_index[j]
                # measure the distance
                for h in range(k - k_cluster, k):
                    if cuttree[i, h] != cuttree[j, h]:
                        pass
                    else:
                        dist = h - (k - k_cluster)
                        hierarchy_dist_matrix[u][v]['weight'] = dist
                        hierarchy_dist_matrix[v][u]['weight'] = dist

        # plot dendrogram
        #plt.figure()
        #dn = hierarchy.dendrogram(Z, labels=resource_index,
        #        color_threshold=(k - k_cluster))
        #plt.show()
        #mpld3.show(open_browser=False)

        print('{} organizational entities extracted.'.format(len(entities)))
        print('Distance matrix (based on hierarchy) extracted as GraphML.')
        #TODO
        G = nx.DiGraph(hierarchy_dist_matrix)
        nx.write_graphml(G, 'dist_hierarchy.graphml', encoding='windows-1252') 
        return copy.deepcopy(entities)

#TODO
def complete_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - complete linkage (Farthest Point):')
    pass

#TODO
def average_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - average linkage (UPGMA):')
    pass

#TODO
def weighted_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - weighted linkage (WPGMA):')
    pass

#TODO
# EuclideanDistance only
def centroid_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - centroid linkage (UPGMC):')
    pass

#TODO
# EuclideanDistance only
def median_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - median linkage (WPGMC):')
    pass

#TODO
# EuclideanDistance only
def ward_linkage(cases, k_cluster):
    print('Applying Hierarchical Clustering - Ward linkage (incremental):')
    pass


