#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from orgminer.IO.reader import read_disco_csv
from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    with open(filename_input, 'r') as f:
        el = read_disco_csv(f)

    num_total_cases = len(set(el['case_id']))
    num_total_resources = len(set(el['resource']))

    mode_miner = ATonlyMiner(el)

    # derive resource log
    rl = mode_miner.derive_resource_log(el)

    profiles = count_execution_frequency(rl)
    print(profiles)

