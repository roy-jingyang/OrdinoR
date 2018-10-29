#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Build organizational group network using information flow. 

import csv
import sys
import networkx as nx
import numpy as np
from collections import defaultdict

f_event_log = sys.argv[1]
f_out_network = sys.argv[2]

if __name__ == '__main__':
    '''
    # read organizational model
    org_model = defaultdict(lambda: set())
    with open(f_org_model, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for rid in row[2].split(';'):
                    org_model[row[0]].add(rid)
    '''

    mat_info_flow = defaultdict(lambda: defaultdict(lambda: {'weight': 0.0}))
    #mat_info_flow = defaultdict(lambda: defaultdict(lambda: 0.0))

    cases = defaultdict(lambda: list())
    with open(f_event_log, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0]
                activity = row[1]
                resource = row[2]
                #cases[caseid].append((caseid, activity, resource))
                # the 'org:group' field content should be implicit in fact

                #org_group = row[-1] # wabo
                #org_group = row[10] # bpic2013_open
                org_group = row[9] # bpic2013_closed
                cases[caseid].append((caseid, activity, resource, org_group))
    
    # scale_factor: SIGMA_Case c in Log (|c| - 1)
    scale_factor = 0
    for caseid, trace in cases.items():
        scale_factor += len(trace) - 1

    G = nx.DiGraph()
    for caseid, trace in cases.items():
        for i in range(len(trace) - 1):
            # within a case
            group_prev = trace[i][-1]
            group_next = trace[i + 1][-1]
            mat_info_flow[group_prev][group_next]['weight'] += 1 / scale_factor
            #mat_info_flow[group_prev][group_next] += 1 / scale_factor
    ''' 
    for ix, iys in mat_info_flow.items():
        for iy, wt in iys.items():
            G.add_edge(ix, iy, weight=wt)
    ''' 

    G = nx.from_dict_of_dicts(mat_info_flow)
    print(G.is_directed())
    nx.write_graphml(G, f_out_network)

