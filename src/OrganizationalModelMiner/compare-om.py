#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict
import math

f_org_model = sys.argv[1]
f_org_model_reference = sys.argv[2]

if __name__ == '__main__':
    # read target model as input
    model = defaultdict(lambda: set())
    with open(f_org_model, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for rid in row[2].split(';'):
                    model[row[0]].add(rid)
    # read reference model as input
    ref_model = defaultdict(lambda: set())
    with open(f_org_model_reference, 'r') as f:
        is_header_line = True
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                for rid in row[2].split(';'):
                    ref_model[row[0]].add(rid)

    # compare model using entropy measure
    n_model_resource = sum([len(x) for k, x in model.items()])
    n_ref_model_resource = sum([len(x) for k, x in ref_model.items()])
    if n_model_resource != n_ref_model_resource:
        print('Error: Total #resources of the comparing models do not match.')
        exit(1)
    else:
        model_entropy = 0
        for entity_id_i, entity_i in model.items():
            p_i = len(entity_i) / n_model_resource
            entity_entropy_i = 0 
            for entity_id_j, entity_j in ref_model.items():
                n_ij = len(entity_i.intersection(entity_j))
                n_i  = len(entity_i)
                p_ij = n_ij / n_i
                entity_entropy_i += (-1) * p_ij * math.log2(p_ij)
            model_entropy += p_i * entity_entropy_i
    print('The total entropy of the loaded model {} compared to the reference
    model {} is e = {}'.format(model_entropy))
