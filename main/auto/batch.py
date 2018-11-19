#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess

if __name__ == '__main__':
    fn_setup = sys.argv[1]
    dirout = sys.argv[2]
    switch_parallel = len(sys.argv) > 3 and int(sys.argv[3]) == 1

    from networkx import freeze, read_graphml, all_simple_paths
    from multiprocessing import Pool
    setup = freeze(read_graphml(fn_setup))
    test_instances = list(path
            for path in all_simple_paths(setup, source='0', target='13'))

    def run_process(path):
        subprocess.check_call(['python', 'main/auto/flow_eval.py', 
            fn_setup, dirout, ','.join(path[1:-1])])


    # NOTE: if time performance are to be tested, do NOT use multi-processing
    if switch_parallel:
        print('Running in parallel mode.')
        # Multi-processes version
        with Pool(len(test_instances)) as p:
            p.map(run_process, test_instances)
    else:
        print('Running in single process mode.')
        # Single-process version
        list(map(run_process, test_instances))

