#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')
from csv import reader
from collections import defaultdict
from numpy import mean, std

from IO.reader import read_org_model_csv_old

fn_org_model = sys.argv[1]
fn_org_ref_model = sys.argv[2] if len(sys.argv) >= 3 else None

if __name__ == '__main__':
    # read target model as input
    model = read_org_model_csv_old(fn_org_model)

    # read reference model as input
    ref_model = read_org_model_csv_old(fn_org_ref_model)

    # compare model using entropy measure
    model_resource = set()
    model_resource_labels = defaultdict(lambda: set())
    for k, x in model.items():
        model_resource = model_resource.union(x)
        for r in x:
            model_resource_labels[r].add(k)
    n_model_resource = len(model_resource)

    if fn_org_ref_model:
        ref_model_resource = set()
        ref_model_resource_labels = defaultdict(lambda: set())
        for k, y in ref_model.items():
            ref_model_resource = ref_model_resource.union(y)
            for r in y:
                ref_model_resource_labels[r].add(k)
        n_ref_model_resource = len(ref_model_resource)

        if n_model_resource != n_ref_model_resource:
            print('Error: Total #resources of the comparing models do not match:')
            print('#resources in target model = {}'.format(n_model_resource))
            print('#resources in reference model = {}'.format(n_ref_model_resource))
            exit(1)
        else:
            print('\n')
            print('#groups in the target model "{}"'.format(
                fn_org_model), end='')
            print('\n\t\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
                len(model), 
                mean([len(x) for k, x in model.items()]),
                std([len(x) for k, x in model.items()])))
            print('\t\tEach resource belongs to {:.3f} groups on avg'.format(
                mean([len(l) for r, l in model_resource_labels.items()])))

            print('#groups in the reference model "{}"'.format(
                fn_org_ref_model), end='')
            print('\n\t\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
                len(ref_model), 
                mean([len(y) for k, y in ref_model.items()]),
                std([len(y) for k, y in ref_model.items()])))
            print('\t\tEach resource belongs to {:.3f} groups on avg'.format(
                mean([len(l) for r, l in ref_model_resource_labels.items()])))

            print('\n')

            from Evaluation.m2m.cluster_comparison import (
                    report_set_matching, report_counting_pairs,
                    report_entropy_based, report_BCubed_metrics)

            #report_set_matching(model_resource, model, ref_model)
            #report_counting_pairs(model_resource, model, ref_model)
            #report_entropy_based(model_resource, model, ref_model)
            report_BCubed_metrics(model_resource, model, ref_model)
    else:
        print('\n')
        print('#groups in the model "{}"'.format(fn_org_model), end='')
        print('\n\t{} with {:.1f} resources on avg (SD = {:.3f})'.format(
            len(model), 
            mean([len(x) for k, x in model.items()]),
            std([len(x) for k, x in model.items()])))
        print('\tEach resource belongs to {:.3f} groups on avg'.format(
            mean([len(l) for r, l in model_resource_labels.items()])))

        print('-' * 80)

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
                exit('Fatal error: resource possessing invalid membership.')

        print('{}/{} ({:.2%}) resources having multiple membership:'.format(
            cnt_multi, len(model_resource_labels),
            cnt_multi / len(model_resource_labels)))
        #for r in sorted(res_multi_membership):
        #    print(r)
        print(sorted(res_multi_membership))

        print('\n{}/{} ({:.2%}) resources having single membership:'.format(
            cnt_single, len(model_resource_labels),
            cnt_single / len(model_resource_labels)))
        print(sorted([r for r in model_resource if r not in
            res_multi_membership]))

        print('-' * 80)

        model_keys = sorted(list(model.keys()))
        for i in range(len(model_keys) - 1):
            for j in range(i + 1, len(model_keys)):
                if model[model_keys[i]] < model[model_keys[j]]:
                    print('"{}" is a proper subset of "{}"!'.format(
                        model_keys[i], model_keys[j]))
                if model[model_keys[j]] < model[model_keys[i]]:
                    print('"{}" is a proper subset of "{}"!'.format(
                        model_keys[j], model_keys[i]))

