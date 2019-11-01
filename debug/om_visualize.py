#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: This script is to be incorporated to the main programs.

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fn_org_model = sys.argv[2]
fnout_graph = sys.argv[3]

def plot_clustering(X_red, labels):
    import numpy as np
    from numpy import min, max
    x_min, x_max = min(X_red, axis=0), max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=X_red[:,0], y=X_red[:,1], c=labels)

if __name__ == '__main__':
    from OrganizationalModelMiner.base import OrganizationalModel
    import matplotlib
    matplotlib.use("Agg") 
    from matplotlib import pyplot as plt
    from sklearn.manifold import SpectralEmbedding
    from scipy.spatial.distance import pdist, squareform
    from numpy import array, mean

    # read organizational model
    with open(fn_org_model, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)

    # rebuild the resource profiles 
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={'(case) channel': 6})

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.direct_groupby import ATonlyMiner
    from ExecutionModeMiner.direct_groupby import FullMiner
    from ExecutionModeMiner.informed_groupby import TraceClusteringFullMiner

    mode_miner = ATonlyMiner(el)
    #mode_miner = FullMiner(el, 
    #    case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/wabo.bosek5.tcreport', resolution='weekday')

    rl = mode_miner.derive_resource_log(el)
    # build profiles
    from ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl, scale='normalize')

    resources = list(profiles.index)
    # calculate the centroids
    centroids = list()
    for og in om.find_all_groups():
        centroids.append(mean(profiles.loc[list(og)].values, axis=0))

    labels = array(range(om.size()))
    
    # plotting
    # project to 2-d plane
    X_red = SpectralEmbedding(n_components=2, affinity=(
        lambda X: squareform(pdist(X, metric='euclidean')))).fit_transform(
            array(centroids))
    plot_clustering(X_red, labels)
    print('{} groups plotted (as centroids)'.format(om.size()))
    plt.savefig(fnout_graph) 

