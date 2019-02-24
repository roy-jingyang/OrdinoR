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
        el = reader(f, mapping=params.get('mapping', None))

    # Step 1: define execution modes
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

    # Step 2: characterizing resources
    step += 1
    profiler = _import_block(sequence[step]['invoke'])
    params = sequence[step].get('params', None)
    if params is None:
        exit('[Node Error]\t"{}"'.format(sequence[step]['label']))
    else:
        params = eval(params)
    profiles = profiler(rl, **params)

    # Step 3: discover resource grouping
    step += 1
    discoverer = _import_block(sequence[step]['invoke'])
    discoverer_name = sequence[step]['label'].replace(' ', '')
    params = sequence[step].get('params', None)
    if params is None:
        exit('[Node Error]\t"{}"'.format(sequence[step]['label']))
    else:
        params = eval(params)
    ogs = discoverer(profiles, **params)
    if type(ogs) is tuple:
        ogs = ogs[0]

    # TODO: Hard-coded evalution measure (TBD)
    # `. Intrinsic evaluation of clustering (by Silhouette score)
    from Evaluation.m2m.cluster_validation import silhouette_score
    silhouette = silhouette_score(ogs, profiles)

    # assign execution modes
    from OrganizationalModelMiner.base import OrganizationalModel
    om = OrganizationalModel()
    step += 1
    assigner = _import_block(sequence[step]['invoke'])
    assigner_name = sequence[step]['label'].replace(' ', '')
    for og in ogs:
        params = sequence[step].get('params', None)
        if params is None:
            modes = assigner(og, rl)
        else:
            params = eval(params)
            modes = assigner(og, rl, **params)
        om.add_group(og, modes)

    # evaluate organizational model: fitness
    step += 1
    fitness_eval = _import_block(sequence[step]['invoke'])
    fitness = fitness_eval(rl, om)

    # evaluate organizational model: precision
    step += 1
    precision_eval = _import_block(sequence[step]['invoke'])
    precision = precision_eval(rl, om)

    # TODO: Hard-coded evalution measure (TBD) cont.
    # 2. (New) Fitness & Precision values
    from Evaluation.l2m.conformance import fitness1, precision2, precision3
    fitness1 = fitness1(rl, om)
    precision2 = precision2(rl, om)
    precision3 = precision3(rl, om)

    # 3. Overlapping Density & Overlapping Diversity (avg.)
    k = om.size()
    resources = om.resources()
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
    fnout = discoverer_name + '.om'
    with open(join(exp_dirpath, fnout), 'w') as fout:
        om.to_file_csv(fout)
    '''

    #return exec_mode_miner_name, k, fitness, precision
    #return discoverer_name, k, fitness, precision
    #return assigner_name, k, fitness, precision
    return ('{}-{}'.format(discoverer_name, assigner_name), 
            k, fitness, precision, 
            fitness1, precision2, precision3,
            ov_density, avg_ov_diversity, silhouette)

if __name__ == '__main__':
    fn_setup = sys.argv[1]
    dirout = sys.argv[2]
    path = sys.argv[3].split(',')

    from networkx import read_graphml
    setup = read_graphml(fn_setup)

    n_tests = 1
    name = ''
    k_values = list()
    fitness_values = list()
    precision_values = list()

    fitness1_values = list()
    precision2_values = list()
    precision3_values = list()
    ov_density_values = list()
    avg_ov_diversity_values = list()
    silhouette_values = list()

    execute_time = list()

    from time import time
    for i in range(n_tests):
        start_time = time()
        name, k, f, p, f1, p2, p3, ovden, avg_ovdiv, sil = execute(
                setup, path, dirout)
        end_time = time()
        k_values.append(k)
        fitness_values.append(f)
        precision_values.append(p)

        fitness1_values.append(f1)
        precision2_values.append(p2)
        precision3_values.append(p3)
        ov_density_values.append(ovden)
        avg_ov_diversity_values.append(avg_ovdiv)
        silhouette_values.append(sil)

        execute_time.append(end_time - start_time)

    with open(join(dirout, '{}_report.csv'.format(name)), 'w+') as fout:
        writer = writer(fout)
        for i in range(n_tests):
            writer.writerow([
                name,
                k_values[i], fitness_values[i], precision_values[i],
                fitness1_values[i], 
                precision2_values[i], precision3_values[i],
                ov_density_values[i], avg_ov_diversity_values[i],
                silhouette_values[i],
                execute_time[i]])
    
