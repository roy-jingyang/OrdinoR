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

        from EvaluationOptions.Supervised import report_set_matching
        from EvaluationOptions.Supervised import report_counting_pairs
        from EvaluationOptions.Supervised import report_entropy_based
        from EvaluationOptions.Supervised import report_BCubed_metrics

        if opt_measure == 'all':
            report_set_matching(model_resource, model, ref_model)
            report_counting_pairs(model_resource, model, ref_model)
            report_entropy_based(model_resource, model, ref_model)
            report_BCubed_metrics(model_resource, model, ref_model)
        elif opt_measure == 'set_matching':
            report_set_matching(model_resource, model, ref_model)
        elif opt_measure == 'counting_pairs':
            report_counting_pairs(model_resource, model, ref_model)
        elif opt_measure == 'entropy_based':
            report_entropy_based(model_resource, model, ref_model)
        elif opt_measure == 'BCubed':
            report_BCubed_metrics(model_resource, model, ref_model)
        else:
            pass
        print('\n')

