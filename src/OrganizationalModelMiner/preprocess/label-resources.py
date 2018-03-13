#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
from collections import defaultdict

f_org_model = sys.argv[1]
fout_membership = sys.argv[2]

if __name__ == '__main__':
    model = defaultdict(lambda: set())
    with open(f_org_model, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for rid in row[2].split(';'):
                    model[row[0]].add(rid)

    model_resource = set()
    model_resource_labels = defaultdict(lambda: set())
    for k, x in model.items():
        model_resource = model_resource.union(x)
        for r in x:
            model_resource_labels[r].add(k)
    print('{} resources in {} groups'.format(
        len(model_resource), len(model)))
    
    with open(fout_membership, 'w') as fout:
        writer = csv.writer(fout)
        #writer.writerow(['resource', 'label(s)'])
        for r, label in model_resource_labels.items():
            #writer.writerow([r, ';'.join(label)])
            writer.writerow([r, len(label)])

