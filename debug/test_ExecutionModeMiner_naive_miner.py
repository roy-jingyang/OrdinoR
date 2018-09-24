#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from IO.reader import read_exec_mode_csv
from IO.writer import write_exec_mode_csv
from ExecutionModeMiner.base import naive_miner

# List input parameters from shell
filename_input = sys.argv[1]
filename_exec_mode_map = sys.argv[2]

if __name__ == '__main__':
    # generate from a log
    el = read_disco_csv(filename_input)
    exec_mode_map = naive_miner(el)

    # read from a file
    #exec_mode_map = read_exec_mode_csv(filename_input)

    print(exec_mode_map.atypes)

    exec_mode_map.convert_event_log(el)

    #write_exec_mode_csv(filename_exec_mode_map, exec_mode_map)

