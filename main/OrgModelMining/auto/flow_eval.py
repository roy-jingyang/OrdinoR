#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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
        - (TBD, other attributes for nice visualization in Gephi)
    * each directed edge represents a possible combination of specific methods
    in the predecessor step and the successor step in the approach.

The setup (graph) is stored in GraphML format, which is XML-based and allows
node attributes as well as convenient visualization features.
"""

import sys

from os.path import join
from csv import writer

def _import_block(path_invoke):
    from importlib import import_module
    module = import_module('.'.join(path_invoke.split('.')[:-1]))
    foo = getattr(module, path_invoke.split('.')[-1])
    return foo

def execute(setup, seq_ix, exp_dirpath):
    sequence = list(setup.nodes[ix] for ix in seq_ix)
    test_name = '-'.join(step['label'] for step in sequence)

    # Step 0: input an event log
    step = 0
    reader = _import_block(sequence[step]['invoke'])
    params = sequence[step].get('params', None)
    if params is None:
        exit('[Node Error]\t"{}"'.format(sequence[step]['label']))
    else:
        params = eval(params)
    with open(params['filepath'], 'r') as f:
        el = reader(f)

    # define execution modes
    step += 1
    cls_exec_mode_miner = _import_block(sequence[step]['invoke'])
    exec_mode_miner_name = sequence[step]['label'].replace(' ', '')
    params = sequence[step].get('params', None)
    if params is None:
        exec_mode_miner = cls_exec_mode_miner(el)
    else:
        params = eval(params)
        exec_mode_miner = cls_exec_mode_miner(el, **params)
    rl = exec_mode_miner.derive_resource_log(el)

    # characterize resources
    step += 1
    profiler = _import_block(sequence[step]['invoke'])
    params = sequence[step].get('params', None)
    if params is None:
        profiles = profiler(rl)
    else:
        params = eval(params)
        profiles = profiler(rl, **params)

    # discover resource grouping
    step += 1
    discoverer = _import_block(sequence[step]['invoke'])
    discoverer_name = sequence[step]['label'].replace(' ', '')
    params = sequence[step].get('params', None)
    if params is None:
        ogs = discoverer(profiles)
    else:
        params = eval(params)
        if 'metric' in params:
            prox_metric = params['metric']
        else:
            prox_metric = None
        ogs = discoverer(profiles, **params)
    if type(ogs) is tuple:
        ogs = ogs[0]

    # assign execution modes
    step += 1
    assigner = _import_block(sequence[step]['invoke'])
    assigner_name = sequence[step]['label'].replace(' ', '')
    params = sequence[step].get('params', None)
    if params is None:
        om = assigner(ogs, rl)
    else:
        params = sequence[step].get('params', None)
        params = eval(params)
        om = assigner(ogs, rl, **params)

    # TODO: Hard-coded evalution measure (TBD)
    # 1. Intrinsic evaluation of clustering (by Silhouette score)
    from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
    from numpy import mean
    if prox_metric is not None:
        silhouette = mean(list(
            silhouette_score(ogs, profiles, metric=prox_metric).values()))
    else:
        silhouette = mean(list(
            silhouette_score(ogs, profiles).values()))

    # TODO: Hard-coded evalution measure (TBD) cont.
    # 2. (New) Fitness & Precision values
    from orgminer.Evaluation.l2m.conformance import fitness, precision
    fitness = fitness(rl, om)
    precision = precision(rl, om)

    k = om.group_number
    '''
    # 3. Overlapping Density & Overlapping Diversity (avg.)
    resources = om.resources
    n_ov_res = 0
    n_ov_res_membership = 0
    for r in resources:
        n_res_membership = len(om.find_groups(r))
        if n_res_membership == 1:
            pass
        else:
            n_ov_res += 1
            n_ov_res_membership += n_res_membership

    ov_density = n_ov_res / len(resources)
    avg_ov_diversity = (n_ov_res_membership / n_ov_res 
            if n_ov_res > 0 else float('nan'))
    '''
    
    # export organizational models
    fnout = '{}-{}-{}.om'.format(
        exec_mode_miner_name, discoverer_name, assigner_name)
    with open(join(exp_dirpath, fnout), 'w') as fout:
        om.to_file_csv(fout)

    return ('{}-{}-{}'.format(
        exec_mode_miner_name, discoverer_name, assigner_name), 
        silhouette, 
        k, fitness, precision)

if __name__ == '__main__':
    fn_setup = sys.argv[1]
    dirout = sys.argv[2]
    path = sys.argv[3].split(',')

    from networkx import read_graphml
    setup = read_graphml(fn_setup)

    n_tests = 1
    name = ''

    l_test_results = list()

    for i in range(n_tests):
        result = list(execute(setup, path, dirout))
        name = result[0]
        l_test_results.append(result[1:])

    with open(join(dirout, '{}_report.csv'.format(name)), 'w+') as fout:
        writer = writer(fout)
        for i in range(n_tests):
            writer.writerow(
                [name] + 
                l_test_results[i])
    
