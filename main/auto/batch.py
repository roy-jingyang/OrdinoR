#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
fn_setup = sys.argv[1]
dirout = sys.argv[2]
import subprocess

def run_process(path):
    subprocess.check_call(['python', 'main/auto/flow_eval.py', 
        fn_setup, dirout, ','.join(path[1:-1])])


if __name__ == '__main__':
    from networkx import freeze, read_graphml, all_simple_paths
    from multiprocessing import Pool
    setup = freeze(read_graphml(fn_setup))
    test_instances = list(path[1:-1] 
            for path in all_simple_paths(setup, source='0', target='13'))
    with Pool(len(test_instances)) as p:
        p.map(run_process, test_instances)

