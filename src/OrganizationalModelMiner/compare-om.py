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
    model_resource = set()
    for k, x in model.items():
        model_resource = model_resource.union(x)
    n_model_resource = len(model_resource)
    ref_model_resource = set()
    for k, y in ref_model.items():
        ref_model_resource = ref_model_resource.union(y)
    n_ref_model_resource = len(ref_model_resource)

    if n_model_resource != n_ref_model_resource:
        print('Error: Total #resources of the comparing models do not match:')
        print('#resources in target model = {}'.format(n_model_resource))
        print('#resources in reference model = {}'.format(n_ref_model_resource))
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
                if n_ij != 0:
                    entity_entropy_i += (-1) * p_ij * math.log2(p_ij)
                else:
                    # lim -> 0
                    entity_entropy_i += 0
            model_entropy += p_i * entity_entropy_i
    print('The total entropy of the loaded model "{}"'.format(f_org_model) +
            ' compared to the reference model "{}"'.format(f_org_model_reference)
            + ' is e = {}'.format(model_entropy))
    print('#entities in the loaded model "{}": \t\t\t\t\t{}'.format(
        f_org_model, len(model)))
    print('#entities in the reference  model "{}": \t\t\t{}'.format(
        f_org_model_reference, len(ref_model)))
