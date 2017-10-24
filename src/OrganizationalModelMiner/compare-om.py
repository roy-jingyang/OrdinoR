#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict
import math
import numpy as np

f_org_model = sys.argv[1]
f_org_model_reference = sys.argv[2]
opt_measure = sys.argv[3]


# Supervised
# classification-oriented metrics
def precision_measure(model, ref_model):
    pass

def recall_measure(model, ref_model):
    pass

def F_measure(model, ref_model):
    pass

def entropy_measure(model, ref_model):
    model_entropy = 0
    for entity_id_i, entity_i in model.items():
        p_i = len(entity_i) / n_model_resource
        entity_entropy_i = 0 
        for entity_id_j, entity_j in ref_model.items():
            n_ij = len(entity_i.intersection(entity_j))
            n_i  = len(entity_i)
            p_ij = n_ij / n_i
            if p_ij != 0:
                entity_entropy_i += (-1) * p_ij * math.log2(p_ij)
            else:
                # lim -> 0
                entity_entropy_i += 0
        model_entropy += p_i * entity_entropy_i

    string = 'The total entropy of the loaded model compared to' + \
            'the reference model is e = {}'
    return (model_entropy, string)

def purity_measure(model, ref_model):
    model_purity = 0
    for entity_id_i, entity_i in model.items():
        p_i = len(entity_i) / n_model_resource
        entity_purity_i = 0 
        for entity_id_j, entity_j in ref_model.items():
            n_ij = len(entity_i.intersection(entity_j))
            n_i  = len(entity_i)
            p_ij = n_ij / n_i
            entity_purity_i = p_ij if p_ij > entity_purity_i else entity_purity_i
        model_purity += p_i * entity_purity_i

    string = 'The total purity of the loaded model compared to' + \
            'the reference model is e = {}'
    return (model_purity, string)

# similarity-oriented metrics
def similarity_matrix_metrics(resources, model, ref_model):
    # calculate the matrices
    resources = list(resources)
    n = len(resources)
    mat_cluster = np.eye(n)
    for entity_id, entity in model.items():
        entity = list(entity)
        for i in range(len(entity)):
            for j in range(len(entity)):
                if i != j:
                    u = resources.index(entity[i])
                    v = resources.index(entity[j])
                    mat_cluster[u][v] = 1
                else:
                    pass
    mat_class = np.eye(n)
    for entity_id, entity in ref_model.items():
        entity = list(entity)
        for i in range(len(entity)):
            for j in range(len(entity)):
                if i != j:
                    u = resources.index(entity[i])
                    v = resources.index(entity[j])
                    mat_class[u][v] = 1
                else:
                    pass

    # obtain the correlation
    correlation = -1
    # count from the triangular matrix
    f00 = f01 = f10 = f11 = 0
    for i in range(n):
        for j in range(i + 1, n):
            if mat_class[i][j] == 0 and mat_cluster[i][j] == 0:
                f00 += 1
            elif mat_class[i][j] == 0 and mat_cluster[i][j] == 1:
                f01 += 1
            elif mat_class[i][j] == 1 and mat_cluster[i][j] == 0:
                f10 += 1
            else:
                f11 += 1
    
    Rand_stat = (f00 + f11) / (f00 + f01 + f10 + f11)
    Jaccard_index = f11 / (f01 + f10 + f11)
    string = 'The metrics based on similarity matrix of the loaded model' + \
            'compared to the reference model is: ' + \
            'Corr = {}, Rand_stat = {}, Jaccard_index = {}'
    return (correlation, Rand_stat, Jaccard_index, string)
'''
'''

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
        if opt_measure == 'all':
            print('Supervised evaluation:')
            print('\tClassification-oriented:', end='\n\t\t')
            value, string = entropy_measure(model, ref_model)
            print(string.format(value), end='\n\t\t')
            value, string = purity_measure(model, ref_model)
            print(string.format(value))
            print('\tSimilarity-oriented:', end='\n\t\t')
            value_corr, value_Rand, value_ji, string = \
                    similarity_matrix_metrics(model_resource, model, ref_model)
            print(string.format(value_corr, value_Rand, value_ji))
        elif opt_measure == 'entropy':
            value, string = entropy_measure(model, ref_model)
            print(string.format(value))
        elif opt_measure == 'purity':
            value, string = purity_measure(model, ref_model)
            print(string.format(value))
        elif opt_measure == 'similarity':
            value_corr, value_Rand, value_ji, string = \
                    similarity_matrix_metrics(model_resource, model, ref_model)
            print(string.format(value_corr, value_Rand, value_ji))
        else:
            pass
        print('\n')

