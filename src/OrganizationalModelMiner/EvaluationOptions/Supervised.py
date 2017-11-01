#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np

# Supervised
# classification-oriented metrics
def entropy_measure(resources, model, ref_model):
    n = len(resources)
    model_entropy = 0
    for entity_id_i, entity_i in model.items():
        p_i = len(entity_i) / n
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

def conditional_entropy_measure(resources, model, ref_model):
    # calculate the matrices
    resources = list(resources)
    n = len(resources)

    labels_pred = np.empty((n, 1))
    for entity_id, entity in model.items():
        entity = list(entity)
        for i in range(len(entity)):
            u = resources.index(entity[i])
            labels_pred[u] = list(model.keys()).index(entity_id)

    labels_true = np.empty((n, 1))
    for entity_id, entity in ref_model.items():
        entity = list(entity)
        for i in range(len(entity)):
            u = resources.index(entity[i])
            labels_true[u] = list(ref_model.keys()).index(entity_id)

    from sklearn.metrics import homogeneity_score, completeness_score
    h_score = homogeneity_score(labels_true.ravel(), labels_pred.ravel())
    c_score = completeness_score(labels_true.ravel(), labels_pred.ravel())
    from sklearn.metrics import v_measure_score
    v_score = v_measure_score(labels_true.ravel(), labels_pred.ravel())
    string = 'The metrics based on conditional entropy of the loaded model' + \
            ' compared to the reference model is:\n' + \
            'homogeneity = {}, completeness = {}, V-measure = {}'
    return (h_score, c_score, v_score, string)


def purity_measure(resources, model, ref_model):
    n = len(resources)
    model_purity = 0
    for entity_id_i, entity_i in model.items():
        p_i = len(entity_i) / n
        entity_purity_i = 0 
        for entity_id_j, entity_j in ref_model.items():
            n_ij = len(entity_i.intersection(entity_j))
            n_i  = len(entity_i)
            p_ij = n_ij / n_i
            entity_purity_i = p_ij if p_ij > entity_purity_i else entity_purity_i
        model_purity += p_i * entity_purity_i

    string = 'The total purity of the loaded model compared to' + \
            ' the reference model is e = {}'
    return (model_purity, string)

# similarity-oriented metrics
def similarity_matrix_metrics(resources, model, ref_model):
    print('Warning: The similarity matrix metrics may only be used on' + 
            ' disjoint clustering results.')
    # calculate the matrices
    resources = list(resources)
    n = len(resources)

    mat_cluster = np.eye(n)
    labels_pred = np.empty((n, 1))
    for entity_id, entity in model.items():
        entity = list(entity)
        for i in range(len(entity)):
            u = resources.index(entity[i])
            labels_pred[u] = list(model.keys()).index(entity_id)
            for j in range(len(entity)):
                if i != j:
                    v = resources.index(entity[j])
                    mat_cluster[u][v] = 1

    mat_class = np.eye(n)
    labels_true = np.empty((n, 1))
    for entity_id, entity in ref_model.items():
        entity = list(entity)
        for i in range(len(entity)):
            u = resources.index(entity[i])
            labels_true[u] = list(ref_model.keys()).index(entity_id)
            for j in range(len(entity)):
                if i != j:
                    v = resources.index(entity[j])
                    mat_class[u][v] = 1

    # obtain the correlation
    from scipy.stats import pearsonr
    correlation, p = pearsonr(mat_cluster.flatten(), mat_class.flatten())
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
    from sklearn.metrics import adjusted_rand_score
    ARI = adjusted_rand_score(labels_true.ravel(), labels_pred.ravel())
    Jaccard_index = f11 / (f01 + f10 + f11)
    string = 'The metrics based on similarity matrix of the loaded model' + \
            ' compared to the reference model is:\n' + \
            'Corr = {} (with p-value={}), ' + \
            'Rand_stat = {} (ARI = {}), Jaccard_index = {}'
    return (correlation, p, Rand_stat, ARI, Jaccard_index, string)

