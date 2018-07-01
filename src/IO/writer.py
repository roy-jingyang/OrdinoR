#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module contains methods for exporting models/results.
'''

import sys
import csv

# 1. OrganizationModelMiner-related
def _describe_om(og):
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
