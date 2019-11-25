#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: This script is to be built as an individual module for orgminer.

import sys
sys.path.append('./')

import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt

from numpy import array, mean, min, max, concatenate
from pandas import DataFrame

import seaborn as sns

fn_event_log = sys.argv[1]
fn_org_model = sys.argv[2]
dirout_graph = sys.argv[3]

def normalize(x):
    x_min = min(x)
    x_max = max(x)
    return (x - x_min) / (x_max - x_min)

def plot_clustering_2d(clustering):
    from matplotlib.offsetbox import TextArea, AnnotationBbox

    import sklearn.manifold as manifold
    N_components = 2
    methods = {
        #'Isomap': {}, 
        'MDS': {'metric': True},
    }

    for i, method in enumerate(methods.keys()):
        # transform to 2d
        class_name = method.split('-')[0]
        Method = getattr(manifold, class_name)

        mf = Method(n_components=2, **methods[method]).fit(
            array(clustering['xy_centroids']))
        xy_c_red = mf.embedding_

        df = {
            'Group': list('G{}'.format(x) 
                for x in clustering['labels_centroids']),
            'x_c_red': xy_c_red[:, 0],
            'y_c_red': xy_c_red[:, 1],
            'size': clustering['size'],
        }

        df = DataFrame.from_dict(df)

        sns.set(font_scale=0.5)
        ax = sns.scatterplot(data=df,
            x='x_c_red', y='y_c_red',
            hue='Group', size='size',
            sizes=(10, 200),
            legend=False)

        description_template = '''
            Group #{} with {} resources:
            Capabilities (Top-3 most-related):
            {}'''

        for ig, g in enumerate(clustering['labels_centroids']):
            description = description_template.format(
                g, clustering['size'][ig],
                #','.join(list((clustering['members'][ig]))),
                '\n'.join(str(x) for x in clustering['modes'][ig][:3]))
            #ax.text(df    

        plt.tight_layout()
        fig = ax.get_figure()
        fig.savefig(dirout_graph + '/' + method, dpi=300) 

def plot_clustering_heatmap(clustering, labeled_by='object'):
    df = DataFrame(clustering['xy'], index=clustering['index'])
    df['labels'] = clustering['labels']
    df.sort_values('labels', inplace=True)

    from scipy.spatial.distance import pdist, squareform
    mat_pdist = squareform(pdist(df.values, metric='euclidean'))
    mat_pdist = normalize(mat_pdist)

    #sns.set(font_scale=0.5)

    from matplotlib.patches import Rectangle
    if labeled_by == 'object':
        patch_pos = [0]

        for i, s in enumerate(clustering['size']):
            if i >= 1:
                patch_pos.append(patch_pos[i-1] + clustering['size'][i-1])

        ax = sns.heatmap(data=mat_pdist,
            cmap='vlag',
            xticklabels=df.index,
            yticklabels=df.index)

        # annotate groups
        for i, s in enumerate(clustering['size']):
            pos = patch_pos[i]
            ax.add_patch(Rectangle((pos, pos),
                width=s, height=s,
                fill=False, edgecolor='yellow', 
                linewidth=1))

    elif labeled_by == 'cluster':
        new_labels = list()
        patch_pos = [0]

        for i, s in enumerate(clustering['size']):
            new_labels.append('Group {} (size {})'.format(
                clustering['labels_centroids'][i], s))
            if i >= 1:
                patch_pos.append(patch_pos[i-1] + clustering['size'][i-1])
            for n in range(s - 1):
                new_labels.append('')

        # create a copy for modification
        from copy import deepcopy
        from itertools import permutations
        mat_pdist_md = deepcopy(mat_pdist)
        # step-1: fill cells within clusters
        for i, s in enumerate(clustering['size']):
            pos = patch_pos[i]
            if s > 1:
                new_value = mean(mat_pdist[pos:pos+s, pos:pos+s])
                mat_pdist_md[pos:pos+s, pos:pos+s] = new_value

        # step-2: fill cells between clusters
        l = list()
        for i, s in enumerate(clustering['size']):
            pos = patch_pos[i]
            if i == 0:
                continue

            for prev_i in range(i):
                u = pos
                v = patch_pos[prev_i]
                s_x = clustering['size'][prev_i]
                new_value = mean(mat_pdist[u:u+s, v:v+s_x])
                mat_pdist_md[u:u+s, v:v+s_x] = new_value
                mat_pdist_md[v:v+s_x, u:u+s] = new_value

        #print(len(l))

        ax = sns.heatmap(data=mat_pdist_md,
            cmap='vlag',
            xticklabels=new_labels,
            yticklabels=new_labels)

        # annotate groups
        for i, s in enumerate(clustering['size']):
            pos = patch_pos[i]
            ax.add_patch(Rectangle((pos, pos),
                width=s, height=s,
                fill=False, edgecolor='yellow', 
                linewidth=1))
    else:
        exit('[Error] Unrecognized labelling option.')
        
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(dirout_graph + '/' + 'heatmap', dpi=300)

if __name__ == '__main__':
    from orgminer.OrganizationalModelMiner.base import OrganizationalModel

    # read organizational model
    with open(fn_org_model, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)

    # rebuild the resource profiles 
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={'(case) channel': 6})

    # learn execution modes and convert to resource log
    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
    from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringFullMiner

    #mode_miner = ATonlyMiner(el)
    mode_miner = FullMiner(el, 
        case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/wabo.bosek5.tcreport', resolution='weekday')

    rl = mode_miner.derive_resource_log(el)
    # build profiles
    from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
    profiles = count_execution_frequency(rl)

    # calculate the centroids
    resources = list(profiles.index)
    labels = [-1] * len(resources)

    centroids = list()
    centroid_labels = list()

    group_members = list()
    group_modes = list()
    group_size = list()
    num = 0
    for og_id, og in om.find_all_groups():
        for r in og:
            labels[resources.index(r)] = og_id

        centroids.append(mean(profiles.loc[list(og)].values, axis=0))
        centroid_labels.append(og_id)  

        group_members.append(og)
        group_modes.append(om.find_group_execution_modes(og_id))
        group_size.append(len(og))

    #plot_clustering_2d({
    plot_clustering_heatmap({
        'index': list(profiles.index),
        'xy': profiles.values,
        'labels': labels,
        'xy_centroids': centroids,
        'labels_centroids': centroid_labels,
        'size': group_size,
        'members': group_members,
        'modes': group_modes
        },
        labeled_by='object'
    )

    print('Plotting exported.')

