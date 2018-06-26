#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of hierarchical organizational mining
methods, based on the use of clustering techniques. Methods include:
    1. Agglomerative Hierarchical Clustering (Song & van der Aalst)
'''

import copy
import networkx as nx
import numpy as np

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3

from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy

def ahc(profiles, n_groups, linkage='single'):
    '''
    This method applies the basic agglomerative hierarchical clustering
    algorithm in clustering analysis, proposed by Song & van der Aalst.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
    Returns:
        hcy: DataFrame
            The hierarchical structure as a pandas DataFrame, with resource ids
            as indices, levels of the hierarchy as columns, and group ids as
            the values, e.g. for 20 resources placed in a 5-level hierarhical
            structure with 8 groups at the lowest level, there should be 20
            rows and 5 columns in the DataFrame, and the values should be in
            range of 0 to 7.
    '''
    # TODO
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

