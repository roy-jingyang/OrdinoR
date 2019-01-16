#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from ExecutionModeMiner.naive_miner import ATonlyMiner
from ExecutionModeMiner.naive_miner import ATCTMiner

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    # generate from a log
    with open(filename_input, 'r') as f:
        el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'(case) group': 8})
    naive_exec_mode_miner = ATonlyMiner(el)
    #naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) group')

    # derive resource log
    rl = naive_exec_mode_miner.derive_resource_log(el)
    print(rl)

