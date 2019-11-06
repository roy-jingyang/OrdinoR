#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from orgminer.IO.reader import read_disco_csv
from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringFullMiner

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    # generate from a log
    with open(filename_input, 'r') as f:
        el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'(case) channel': 6})

    mode_miner = ATonlyMiner(el)
    #mode_miner = FullMiner(el, 
    #    case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/bpic12.bosek5.tcreport', resolution='weekday')

    # derive resource log
    rl = mode_miner.derive_resource_log(el)
    print(rl[['case_type', 'activity_type', 'time_type']].drop_duplicates())
    print('Num. = {}'.format(len(
        rl[['case_type', 'activity_type', 'time_type']].drop_duplicates())))

