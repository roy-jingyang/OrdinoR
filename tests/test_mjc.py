#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# import methods to be tested below


# List input parameters from shell
fn_event_log = sys.argv[1]
fnout_network = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)
    # event log preprocessing
    # NOTE: filter cases done by one single resources
    num_total_cases = len(set(el['case_id']))
    num_total_resources = len(set(el['resource']))
    teamwork_cases = set()
    for case_id, events in el.groupby('case_id'):
        if len(set(events['resource'])) > 1:
            teamwork_cases.add(case_id)
    # NOTE: filter resources with low event frequencies (< 1%)
    num_total_events = len(el)
    active_resources = set()
    for resource, events in el.groupby('resource'):
        if (len(events) / num_total_events) >= 0.01:
            active_resources.add(resource)

    el = el.loc[el['resource'].isin(active_resources) 
                & el['case_id'].isin(teamwork_cases)]
    print('{}/{} resources found active in {} cases.\n'.format(
        len(active_resources), num_total_resources,
        len(set(el['case_id']))))
    
    from orgminer.SocialNetworkMiner.joint_cases import working_together
    sn = working_together(el)
    from networkx import write_gexf
    sn = write_gexf(sn, fnout_network)
    '''
    from orgminer.OrganizationalModelMiner.community.graph_partitioning import (
        _mjc)
    for k in range(2, 10):
        ogs, _ = _mjc(el, k, method='centrality')
        if len(ogs) == 0:
            print('Invalid')
        else:
            print('Org. groups discovered with {} groups and {} resources.'.format(
                len(ogs), sum(len(x) for x in ogs)))
    '''

