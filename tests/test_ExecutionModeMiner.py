#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from orgminer.IO.reader import read_disco_csv
from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
from orgminer.ExecutionModeMiner.direct_groupby import CTonlyMiner
from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
from orgminer.ExecutionModeMiner.direct_groupby import ATTTMiner
from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringCTMiner
from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringFullMiner

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    # generate from a log
    with open(filename_input, 'r') as f:
        #el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'(case) channel': 6})
        el = read_disco_csv(f, mapping={'(case) last_phase': 14}) # bpic15-*

    '''
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
    '''

    #mode_miner = ATonlyMiner(el)
    mode_miner = CTonlyMiner(el, case_attr_name='(case) last_phase')
    #mode_miner = ATCTMiner(el, case_attr_name='(case) channel')
    #mode_miner = ATTTMiner(el, resolution='weekday')
    #mode_miner = FullMiner(el, 
    #    case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringCTMiner(el,
    #    fn_partition='input/extra_knowledge/wabo.bosek5.tcreport')
    #mode_miner = TraceClusteringFullMiner(el, 
    #    fn_partition='input/extra_knowledge/wabo.bosek5.tcreport', 
    #    resolution='weekday')

    # derive resource log
    rl = mode_miner.derive_resource_log(el)
    print(rl[['case_type', 'activity_type', 'time_type']].drop_duplicates())
    print('Num. = {}'.format(len(
        rl[['case_type', 'activity_type', 'time_type']].drop_duplicates())))


    '''
    from collections import Counter
    counter = Counter()
    for k, v in mode_miner._ctypes.items():
        counter[v] += 1

    for v, count in counter.items():
        print('{},{},{:.1%}'.format(
            v, count, count / len(mode_miner._ctypes)))
    '''

