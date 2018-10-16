#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program delivers a batch evaluation script for experiments. It allows user
to configure the components to be included as the setup and then save the
setup. The program can then load the setup from the save file, and starts to
instantiate all possible test instances following the setup and exectues.

The underlying data structure of the setup is an unweighted acyclic directed
graph, where:
    * each node represents a step in the whole approach (usually, a method
    from a module should be invoked), and all the configuration and tag
    information related to the method used is stored as node attributes;
    * each node is described by a name specified by the user, and contains the
    following attributes:
        - 'label'
        - 'step'
        - 'invoke'
        - 'params'
        - (TBD, other attributes for nice visualization)
    * each directed edge represents a possible combination of specific methods
    in the predecessor step and the successor step in the approach.

The setup (graph) is stored in GraphML format, which is XML-based and allows
node attributes as well as convenient visualization features.
'''

import sys
sys.path.append('./src/')

def _import_block(path_invoke):
    from importlib import import_module
    module = import_module('.'.join(path_invoke.split('.')[:-1]))
    foo = getattr(module, path_invoke.split('.')[-1])
    return foo

def execute(setup, seq_ix):
    sequence = list(setup.nodes[ix] for ix in seq_ix)
    test_name = ' -> '.join(step['label'] for step in sequence)

    # Step 0: input an event log
    step = 0
    reader = _import_block(sequence[step]['invoke'])
    with open(sequence[step]['params'], 'r') as f:
        el = reader(f)

    # Step 1: define execution modes
    step += 1
    cls_exec_mode_miner = _import_block(sequence[step]['invoke'])
    exec_mode_miner = cls_exec_mode_miner(el)
    rl = exec_mode_miner.derive_resource_log(el)

    # Step 2: characterizing resources
    step += 1
    profiler = _import_block(sequence[step]['invoke'])
    profiles = profiler(rl, **eval(sequence[step]['params']))

    # Step 3: discover resource grouping
    step += 1
    grouping_discoverer = _import_block(sequence[step]['invoke'])
    ogs = grouping_discoverer(profiles, **eval(sequence[step]['params']))

    # assign execution modes
    from OrganizationalModelMiner.base import OrganizationalModel
    om = OrganizationalModel()
    step += 1
    assigner = _import_block(sequence[step]['invoke'])
    for og in ogs:
        om.add_group(og, assigner(og, rl))    

    # evaluate organizational model: fitness
    step += 1
    fitness_eval = _import_block(sequence[step]['invoke'])
    fitness = fitness_eval(rl, om)

    # evaluate organizational model: precision
    step += 1
    precision_eval = _import_block(sequence[step]['invoke'])
    precision = precision_eval(rl, om)

    return test_name, fitness, precision


if __name__ == '__main__':
    fn_setup = sys.argv[1]
    fnout = sys.argv[2]
    path = sys.argv[3].split(',')

    from networkx import read_gexf, read_graphml
    setup = read_graphml(fn_setup)

    name, f, p = execute(setup, path)

    with open(fnout, 'a') as fout:
        fout.write('{}\n'.format(name))
        fout.write('\tFitness   = {:.6f}\n'.format(f))
        fout.write('\tPrecision = {:.6f}\n'.format(p))
    
