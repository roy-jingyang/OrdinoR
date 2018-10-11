#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv

from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner

from OrganizationalModelMiner.base import default_mining

from OrganizationalModelMiner.disjoint import graph_partitioning
from SocialNetworkMiner.joint_activities import distance, correlation
from SocialNetworkMiner.utilities import select_edges_by_weight

from ResourceProfiler.raw_profiler import performer_activity_frequency

from OrganizationalModelMiner.hierarchical import clustering
from OrganizationalModelMiner.hierarchical import community_detection

from OrganizationalModelMiner.overlap.clustering import gmm
from OrganizationalModelMiner.overlap.clustering import moc
from OrganizationalModelMiner.overlap.clustering import fcm

from OrganizationalModelMiner.base import OrganizationalModel

from OrganizationalModelMiner.mode_assignment import member_first_assign

# List input parameters from shell
filename_input = sys.argv[1]
filename_result = sys.argv[2]

if __name__ == '__main__':
    with open(filename_input, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)
        #om = OrganizationalModel.from_file_csv(f)
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    # default mining
    ogs, score = default_mining(rl)

    '''
    # MJA/MJC
    sn = correlation(el, use_log_scale=False)
    sn = select_edges_by_weight(sn, low=0.7235)

    ogs = graph_partitioning.connected_components(sn)
    '''

    '''
    # AHC
    profiles = performer_activity_frequency(rl, use_log_scale=True)
    ogs, og_hcy = clustering.ahc(profiles, 9, method='ward')
    '''

    '''
    # HC (by community detection)
    from SocialNetworkMiner.joint_activities import distance
    sn = distance(el, use_log_scale=True, convert=True)
    ogs, og_hcy = community_detection.betweenness(sn, 9, weight='weight')
    '''

    '''
    # Sent to 'field test':
    # GMM
    # MOC
    # FCM
    # Appice
    '''

    om = OrganizationalModel()
    for og in ogs:
        om.add_group(og, member_first_assign(og, rl))

    with open(filename_result, 'w', encoding='utf-8') as f:
        om.to_file_csv(f)

