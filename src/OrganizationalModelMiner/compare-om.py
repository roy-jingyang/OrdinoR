#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict
from numpy import mean, std

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
    if not (f_org_model_reference == 'none' and opt_measure == 'none'):
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
    model_resource_labels = defaultdict(lambda: set())
    for k, x in model.items():
        model_resource = model_resource.union(x)
        for r in x:
            model_resource_labels[r].add(k)
    n_model_resource = len(model_resource)

    if not (f_org_model_reference == 'none' and opt_measure == 'none'):
        ref_model_resource = set()
        ref_model_resource_labels = defaultdict(lambda: set())
        for k, y in ref_model.items():
            ref_model_resource = ref_model_resource.union(y)
            for r in y:
                ref_model_resource_labels[r].add(k)
        n_ref_model_resource = len(ref_model_resource)

        from numpy import mean
        if n_model_resource != n_ref_model_resource:
            print('Error: Total #resources of the comparing models do not match:')
            print('#resources in target model = {}'.format(n_model_resource))
            print('#resources in reference model = {}'.format(n_ref_model_resource))
            exit(1)
        else:
            print('\n')
            print('#entities in the target model "{}"'.format(
                f_org_model), end='')
            print('\n\t\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
                len(model), 
                mean([len(x) for k, x in model.items()]),
                std([len(x) for k, x in model.items()])))
            print('\t\tEach resource belongs to {:.3f} entities on avg'.format(
                mean([len(l) for r, l in model_resource_labels.items()])))

            print('#entities in the reference model "{}"'.format(
                f_org_model_reference), end='')
            print('\n\t\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
                len(ref_model), 
                mean([len(y) for k, y in ref_model.items()]),
                std([len(y) for k, y in ref_model.items()])))
            print('\t\tEach resource belongs to {:.3f} entities on avg'.format(
                mean([len(l) for r, l in ref_model_resource_labels.items()])))

            print('\n')

            from EvaluationOptions.Supervised import report_set_matching
            from EvaluationOptions.Supervised import report_counting_pairs
            from EvaluationOptions.Supervised import report_entropy_based
            from EvaluationOptions.Supervised import report_BCubed_metrics

            if opt_measure == 'all':
                #report_set_matching(model_resource, model, ref_model)
                #report_counting_pairs(model_resource, model, ref_model)
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
    else:
        print('\n')
        print('#entities in the model "{}"'.format(f_org_model), end='')
        print('\n\t\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
            len(model), 
            mean([len(x) for k, x in model.items()]),
            std([len(x) for k, x in model.items()])))
        print('\t\tEach resource belongs to {:.3f} entities on avg'.format(
            mean([len(l) for r, l in model_resource_labels.items()])))
        cnt_multi = 0
        res_multi_membership = list()
        cnt_single = 0
        for r, l in model_resource_labels.items():
            if len(l) > 1:
                cnt_multi += 1
                res_multi_membership.append(r)
            elif len(l) == 1:
                cnt_single += 1
            else:
                print('Ah!')

        print('{} ({:.2%}) resources having multiple membership'.format(
            cnt_multi, cnt_multi / len(model_resource_labels)))
        for r in sorted(res_multi_membership):
            print(r)
        #print(sorted(res_multi_membership))
        print('{} ({:.2%}) resources having single membership'.format(
            cnt_single, cnt_single / len(model_resource_labels)))

        model_keys = sorted(list(model.keys()))
        for i in range(len(model_keys) - 1):
            for j in range(i + 1, len(model_keys)):
                if model[model_keys[i]] <= model[model_keys[j]]:
                    print('{} is subset of {}!'.format(
                        model_keys[i], model_keys[j]))
                if model[model_keys[j]] <= model[model_keys[i]]:
                    print('{} is subset of {}!'.format(
                        model_keys[j], model_keys[i]))

