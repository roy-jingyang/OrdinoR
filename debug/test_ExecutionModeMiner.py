#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner

# List input parameters from shell
filename_input = sys.argv[1]
filename_exec_mode_map = sys.argv[2]

if __name__ == '__main__':
    # generate from a log
    el = read_disco_csv(filename_input)
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)

    # convert to resource log
    rl = naive_exec_mode_miner.convert_event_log(el)
    print(rl)

