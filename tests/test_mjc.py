#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# import methods to be tested below
from networkx import write_gexf

from ordinor.io import read_disco_csv
import ordinor.constants as const
from ordinor.social_network_miner.joint_cases import working_together

# List input parameters from shell
fn_event_log = sys.argv[1]
fnout_network = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    el = read_disco_csv(fn_event_log)
    # event log preprocessing
    # NOTE: filter cases done by one single resources
    num_total_cases = len(set(el[const.CASE_ID]))
    num_total_resources = len(set(el[const.RESOURCE]))
    teamwork_cases = set()
    for case_id, events in el.groupby(const.CASE_ID):
        if len(set(events[const.RESOURCE])) > 1:
            teamwork_cases.add(case_id)
    # NOTE: filter resources with low event frequencies (< 1%)
    num_total_events = len(el)
    active_resources = set()
    for resource, events in el.groupby(const.RESOURCE):
        if (len(events) / num_total_events) >= 0.01:
            active_resources.add(resource)

    el = el.loc[el[const.RESOURCE].isin(active_resources) 
                & el[const.CASE_ID].isin(teamwork_cases)]
    print('{}/{} resources found active in {} cases.\n'.format(
        len(active_resources), num_total_resources,
        len(set(el[const.CASE_ID]))))
    
    sn = working_together(el)
    sn = write_gexf(sn, fnout_network)
