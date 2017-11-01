#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict

f_org_model = sys.argv[1]
f_org_model_reference = sys.argv[2]
opt_measure = sys.argv[3]

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
        print('\n')
        print('#entities in the loaded model "{}":\n\t\t{}'.format(
            f_org_model, len(model)))
        print('#entities in the reference model "{}":\n\t\t{}'.format(
            f_org_model_reference, len(ref_model)))
        print('\n')

        from EvaluationOptions.Supervised import \
                entropy_measure, conditional_entropy_measure, purity_measure, \
                similarity_matrix_metrics

        if opt_measure == 'all':
            print('Supervised evaluation:')
            print('\tClassification-oriented:', end='\n\t\t')
            value, string = entropy_measure(model_resource, model, ref_model)
            print(string.format(value), end='\n\t\t')
            value_h, value_c, value_v, string = conditional_entropy_measure(
                    model_resource, model, ref_model)
            print(string.format(value_h, value_c, value_v), end='\n\t\t')
            value, string = purity_measure(model_resource, model, ref_model)
            print(string.format(value))
            print('\tSimilarity-oriented:', end='\n\t\t')
            value_corr, value_p, value_Rand, value_ARI, value_ji, string = \
                    similarity_matrix_metrics(model_resource, model, ref_model)
            print(string.format(
                value_corr, value_p,
                value_Rand, value_ARI, value_ji))
        elif opt_measure == 'entropy':
            value, string = entropy_measure(model_resource, model, ref_model)
            print(string.format(value))
        elif opt_measure == 'cond_entropy':
            value_h, value_c, value_v, string = conditional_entropy_measure(
                    model_resource, model, ref_model)
            print(string.format(value_h, value_c, value_v))
        elif opt_measure == 'purity':
            value, string = purity_measure(model_resource, model, ref_model)
            print(string.format(value))
        elif opt_measure == 'similarity':
            value_corr, value_p, value_Rand, value_ARI, value_ji, string = \
                    similarity_matrix_metrics(model_resource, model, ref_model)
            print(string.format(
                value_corr, value_p,
                value_Rand, value_ARI, value_ji))
        else:
            pass
        print('\n')

