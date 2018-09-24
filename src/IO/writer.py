# -*- coding: utf-8 -*-

'''
This module contains methods for exporting models/results.
'''

import sys
import csv

# 0. ExecutionModeMiner-related
# 0.1. Write the corresponding map of execution modes to file as "plain" CSV

def _describe_exec_mode_map(exec_mode_map):
    '''
    Params:
        exec_mode_map: an object of class ExecutionModeMap.
    Returns:
    '''

    print('-' * 80)

    print('Count of Types:')
    print('Number of Case Types:\t\t{}'.format(len(exec_mode_map.ctypes)))
    print('Number of Activity Types:\t{}'.format(len(exec_mode_map.atypes)))
    print('Number of Time Types:\t\t{}'.format(len(exec_mode_map.ttypes)))
        
    print('-' * 80)
    return

def write_exec_mode_csv(fn, exec_mode_map):
    '''
    Params:
        exec_mode_map: an object of class ExecutionModeMap.
    Returns:
    '''

    _describe_exec_mode_map(exec_mode_map)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        for type_name, identifiers in exec_mode_map.ctypes.items():
            writer.writerow([
                    type_name,
                    ';'.join(sorted(identifiers))])
        for type_name, identifiers in exec_mode_map.atypes.items():
            writer.writerow([
                    type_name,
                    ';'.join(sorted(identifiers))])
        for type_name, identifiers in exec_mode_map.ttypes.items():
            writer.writerow([
                    type_name,
                    ';'.join(sorted(identifiers))])
    print('Execution Mode Map exported.')
    return

# 1. OrganizationModelMiner-related
# TODO Rewrite after the class definition of OrgnizationalModel is done.
# This part is merely a wrap-up for the related class methods.

def _describe_om(om):
    '''
    Params:
        og: dict of sets
            The organizational groups discovered.
    Returns:
    '''

    print('-' * 80)

    print('Number of organizational groups:\t\t{}'.format(len(og)))

    print('-' * 80)
    return

# 1.1. Write organizational model to file as "plain" CSV
def write_om_csv(fn, og, a):
    '''
    Params:
        fn: str
            Filename of the file to be exported.
        og: dict of sets
            The organizational groups discovered.
        a: dict of sets
            The entity assignment result.
    Returns:
    '''

    _describe_om(og)
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['group id', 'tasks', 'resources'])
        for gid in og.keys():
            writer.writerow([
                gid,
                ';'.join(sorted(t for t in a[gid])),
                ';'.join(sorted(r for r in og[gid]))
            ])
    print('Organizational model exported.')
    return

# 1.2. Write organizational model to file as OMML format (see DSS 2008)

# 2. SocialNetworkMiner-related 
def _describe_sn(sn):
    '''
    Params:
        sn: NetworX (Di)Graph
            The social network discovered.
    Returns:
    '''

    print('-' * 80)

    print('Number of nodes:\t\t{}'.format(len(sn.nodes)))
    print('Number of edges:\t\t{}'.format(len(sn.edges)))
    from networkx import density
    print('Density:\t\t{}'.format(density(sn)))

    print('-' * 80)
    return

# 2.1 Write social network model to file as GraphML format
def write_sn_graphml(fn, sn):
    '''
    Params:
        fn: str
            Filename of the file to be exported.
        sn: NetworX (Di)Graph
            The social network discovered.
    Returns:
    '''

    from networkx import write_graphml
    _describe_sn(sn)
    write_graphml(sn, fn)
    print('Social network model exported.')
    return

# 3. RuleMiner-related

# 4. Miscellaneous
def write_event_stream(path, c):
    pass

