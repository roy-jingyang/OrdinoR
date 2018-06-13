#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Build organizational group network using information flow. 

import csv
import sys
import networkx as nx
import numpy as np
from collections import defaultdict

f_event_log = sys.argv[1]
f_network = sys.argv[2]

if __name__ == '__main__':
    G = nx.read_graphml(f_network)
    print(len(G.nodes))
    print(len(G.edges))
    hierarchy_dist_matrix = nx.to_dict_of_dicts(G) 

    cases = defaultdict(lambda: list())
    with open(f_event_log, 'r', encoding='windows-1252') as f:
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
                #org_group = row[9] # bpic2013_closed
                #cases[caseid].append((caseid, activity, resource, org_group))
                cases[caseid].append((caseid, activity, resource))

    # process case by case
    pofi_total = 0.0
    for caseid, trace in cases.items():
        for i in range(len(trace) - 1):
            res_prev = trace[i][2]
            res_next = trace[i + 1][2]

            # Transfer of Work Overhead (TWO) based on event log
            # TODO
            if res_prev != res_next:
                pofi_total += hierarchy_dist_matrix[res_prev][res_next]['weight']

    print('The POFI (avg. on #cases) = {}'.format(pofi_total / len(cases)))

