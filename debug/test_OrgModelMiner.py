#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from IO.writer import write_om_csv

from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner

#from OrganizationalModelMiner.base import default_mining

from OrganizationalModelMiner.disjoint import graph_partitioning
from SocialNetworkMiner.joint_activities import distance, correlation
from SocialNetworkMiner.utilities import select_edges_by_weight

# List input parameters from shell
filename_event_log = sys.argv[1]
filename_result = sys.argv[2]

if __name__ == '__main__':
    el = read_disco_csv(filename_event_log)
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.convert_event_log(el)

    #om = default_mining(rl)

    sn = distance(el, use_log_scale=True, convert=True)
    sn = select_edges_by_weight(sn, low=0.9)

    om = graph_partitioning.connected_components(sn, rl)

    write_om_csv(filename_result, om)    

