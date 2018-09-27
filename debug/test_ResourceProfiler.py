#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner
from ResourceProfiler.raw_profiler import performer_activity_frequency

# List input parameters from shell
filename_input = sys.argv[1]
filename_output = sys.argv[2]

if __name__ == '__main__':
    # generate from a log
    el = read_disco_csv(filename_input)
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)

    # derive resource log
    rl = naive_exec_mode_miner.derive_resource_log(el)

    X = performer_activity_frequency(rl, use_log_scale=False)
    print(X)

