#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import networkx as nx

f_org_model = sys.argv[1]
f_sn_model = sys.argv[2]

if __name__ == '__main__':
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
    model_resource = set()
    for k, x in org_model.items():
        model_resource = model_resource.union(x)
    # read social network model
    sn_model = nx.read_gml(f_sn_model)

    print(len(model_resource) == len(sn_model))

