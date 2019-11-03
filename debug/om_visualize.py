#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: This script is to be incorporated to the main programs.

import sys
sys.path.append('./src/')

import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt
from numpy import array, mean, min, max

fn_event_log = sys.argv[1]
fn_org_model = sys.argv[2]
dirout_graph = sys.argv[3]

def plot_clustering_2d(clustering):
    xy = X_dict['x']
    dot_labels = array(list(('Group {}'.format(g)) for g in labels))

    import matplotlib.gridspec as gridspec
    from scipy.spatial.distance import pdist, squareform

    import sklearn.manifold as manifold

    N_components = 2
    methods = {
        'Isomap': {}, 
        'MDS': {'metric': True},
    }

    for i, method in enumerate(methods.keys()):
        fig = plt.figure(figsize=(10, 10), dpi=100)
        #fig(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111) 
        ax.set_title(method)
        class_name = method.split('-')[0]
        Method = getattr(manifold, class_name)
        xy_red = Method(n_components=2, **methods[method]).fit_transform(
            array(X))
        ax.scatter(x=xy_red[:, 0], y=xy_red[:, 1], c=labels,
            s=clustering['size'])
        for ig, g in enumerate(dot_labels):
            ax.annotate(g, (xy[ig, 0], xy[ig, 1]))
        #texts = 
        ax.axis('off')
        plt.savefig(dirout_graph + '/' + method) 


if __name__ == '__main__':
    from OrganizationalModelMiner.base import OrganizationalModel

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

    #mode_miner = ATonlyMiner(el)
    mode_miner = FullMiner(el, 
        case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/wabo.bosek5.tcreport', resolution='weekday')

    rl = mode_miner.derive_resource_log(el)
    # build profiles
    from ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl, scale='normalize')

    '''
    resources = list(profiles.index)
    labels = array([-1] * len(resources))
    for i, og in enumerate(om.find_all_groups()):
        for r in og:
            labels[resources.index(r)] = i

    plot_clustering_2d(profiles.values, labels)
    print('{} resources plotted'.format(len(resources)))
    '''
    # calculate the centroids
    centroids = list()
    group_members = list()
    group_modes = list()
    group_size = list()
    for og_id, og in om.find_all_groups():
        centroids.append(mean(profiles.loc[list(og)].values, axis=0))
        group_members.append(og)
        group_modes.append(om.find_group_execution_modes(og_id))
        group_size.append(len(og))
    centroid_labels = array(range(om.size()))

    print(om._mem[2])
    print(group_members[2])
    print(group_size[2])
    print(om._cap[2])
    print(group_modes[2])
    exit()

    plot_clustering_2d({
        'xy': centroids,
        'label': centroid_labels,
        'size': list(40 * n for n in group_size),
        #'members': 
    })
    print('{} groups plotted (as centroids)'.format(om.size()))

    

