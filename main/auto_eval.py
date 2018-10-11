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
        - 'name'
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

import networkx as nx

fnout_config = sys.argv[1]

def config_setup(fn):
    setup = nx.DiGraph()
    # Add source node (note that this is an 'empty' one)
    setup.add_node('start', **{
        'name': '',
        'step': '',
        'invoke': '',
        'params': ''
    })

    setup.add_node('0.0', **{
        'name': 'Log of receipt phase of WABO',
        'step': 0,
        'invoke': 'IO.reader.read_disco_csv',
        'params': './input/wabo.csv'
    })
    setup.add_edges_from([('start', '0.0')])
    '''
    setup.add_node('0.1', **{
        'name': 'BPiC2013 log (open problems)',
        'step': 0,
        'invoke': 'IO.reader.read_disco_csv',
        'params': './input/bpic2013_open.csv'
    })
    '''
    #setup.add_edges_from([('start', '0.0'), ('start', '0.1')])

    setup.add_node('1.0', **{
        'name': 'Baseline method: use activity names directly as AT',
        'step': 1,
        'invoke': 'ExecutionModeMiner.naive_miner.NaiveActivityNameExecutionModeMiner',
        'params': ''
    })
    setup.add_edges_from([('0.0', '1.0')])

    setup.add_node('2.0', **{
        'name': 'performer-by-activity matrix',
        'step': 2,
        'invoke': 'ResourceProfiler.raw_profiler.performer_activity_frequency',
        'params': {'use_log_scale': True}
    })
    setup.add_edges_from([('1.0', '2.0')])

    setup.add_node('3.0', **{
        'name': 'Model based Overlapping Clustering (MOC)',
        'step': 3,
        'invoke': 'OrganizationalModelMiner.overlap.clustering.moc',
        'params': {'num_groups': 9}
    })
    setup.add_node('3.1', **{
        'name': 'Fuzzy C-Means (FCM)',
        'step': 3,
        'invoke': 'OrganizationalModelMiner.overlap.clustering.fcm',
        'params': {'num_groups': 9}
    })
    setup.add_edges_from([('2.0', '3.0'), ('2.0', '3.1')])

    # Step 4: assign execution modes
    setup.add_node('4.0', **{
        'name': 'member first assign ("any")',
        'step': 4,
        'invoke': 'OrganizationalModelMiner.mode_assignment.member_first_assign',
        'params': ''
    })
    setup.add_node('4.1', **{
        'name': 'group first assign ("all")',
        'step': 4,
        'invoke': 'OrganizationalModelMiner.mode_assignment.member_first_assign',
        'params': ''
    })
    setup.add_edges_from(
            [('3.0', '4.0'), ('3.0', '4.1'), ('3.1', '4.0'), ('3.1', '4.1')])


    # Add source node (note that this is an 'empty' one)
    setup.add_node('end', **{
        'name': '',
        'step': '',
        'invoke': '',
        'params': ''
    })
    setup.add_edges_from(
            [('4.0', 'end'), ('4.1', 'end')])

    # export and return
    print('Configured setup exported to "{}".'.format(fn))
    nx.write_gexf(setup, fn)
    return setup

def _import_block(path_invoke):
    from importlib import import_module
    module = importlib.import_module(path_invoke.split('.')[:-1])
    foo = getattr(path_invoke.split('.')[-1])
    return foo

def execute(setup, seq_ix):
    sequence = list(setup.nodes[ix] for ix in seq_ix)
    test_name = ' -> '.join(step['name'] for step in sequence)
    print('Start test instance:\t{}'.format(test_name))

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
    profiles = profiler(rl, **sequence[step]['params'])

    # Step 3: discover resource grouping
    step += 1
    grouping_discoverer = _import_block(sequence[step]['invoke'])
    ogs = grouping_discoverer(profiles, **sequence[step]['params'])


if __name__ == '__main__':
    '''
    pkg = importlib.import_module('IO.reader')
    foo = getattr(pkg, 'read_disco_csv')
    print(foo)
    with open(sys.argv[1], 'r') as f:
        foo(f)
    '''
    setup = config_setup(fnout_config) 
    test_instances = [path[1:-2] 
            for path in nx.all_simple_paths(setup, source='start', target='end')]

    for test in test_instances:
        test_name = ' -> '.join(setup.nodes[x]['name'] for x in test)
        execute(setup, test)

