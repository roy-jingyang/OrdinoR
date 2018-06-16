#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This module contains methods for exporting models/results.
'''

import sys
import csv

# 1. OrganizationModelMiner-related
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
    with open(fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['group id', 'tasks', 'resources'])
        for gid in og.keys():
            writer.writerow([
                gid,
                ';'.join(sorted(t for t in a[gid])),
                ';'.join(sorted(r for r in og[gid]))
            ])

    return

# 1.2. Write organizational model to file as OMML format (see DSS 2008)
# 2. SocialNetworkMiner-related 
# 2.1 Write social network model to file as GraphML format
# 3. RuleMiner-related

