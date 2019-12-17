#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess

if __name__ == '__main__':
    fn_setup = sys.argv[1]
    dirout = sys.argv[2]

    from networkx import freeze, read_graphml, all_simple_paths
    from multiprocessing import Pool
    setup = freeze(read_graphml(fn_setup))
    test_instances = list(path
        for path in all_simple_paths(setup, source='0', target='13'))
    

    def run_process(path):
        subprocess.check_call(['python', 'main/OrgModelMining/auto/flow_eval.py', 
            fn_setup, dirout, ','.join(path[1:-1])])

    list(map(run_process, test_instances))

