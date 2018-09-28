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

from OrganizationalModelMiner.hierarchical import cluster

from OrganizationalModelMiner.base import OrganizationalModel

from OrganizationalModelMiner.mode_assignment import member_first_assign
from OrganizationalModelMiner.mode_assignment import group_first_assign

from Evaluation.l2m import conformance

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    with open(filename_input, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    # default mining
    ogs = default_mining(rl)

    #sn = correlation(el, use_log_scale=False)
    #sn = select_edges_by_weight(sn, low=0.7)
    #ogs = graph_partitioning.connected_components(sn)

    om = OrganizationalModel()
    for og in ogs:
        #om.add_group(og, member_first_assign(og, rl))
        om.add_group(og, group_first_assign(og, rl))
    
    print()
    print('Fitness = {}'.format(conformance.fitness(rl, om)))
    print('Precision = {}'.format(conformance.precision(rl, om)))

