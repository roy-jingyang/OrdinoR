#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Ref: Amigó, E., Gonzalo, J., Artiles, J., & Verdejo, F. (2009). A comparison
of extrinsic clustering evaluation metrics based on formal constraints.
Information retrieval, 12(4), 461-486.
'''

import math
import numpy as np

# 1. Evaluation by set matching
def report_set_matching(resources, model, ref_model):
    sc_purity, str_purity = purity(resources, model, ref_model)
    sc_inv_purity, str_inv_purity = inverse_purity(resources, model, ref_model)
    sc_F_measure, str_F_measure = F_measure(resources, model, ref_model)
    print('Scoring report of evaluation using "set matching":')
    print('\t', end='')
    print(str_purity.format(sc_purity))
    print('\t', end='')
    print(str_inv_purity.format(sc_inv_purity))
    print('\t', end='')
    print(str_F_measure.format(sc_F_measure))

def purity(resources, model, ref_model):
    N = len(resources)
    model_purity = 0.0
    for entity_id_i, entity_i in model.items(): # cluster Ci
        n_i  = len(entity_i) # |Ci|
        p_i = n_i / N # proportion of Ci
        entity_purity_i = 0.0
        for entity_id_j, entity_j in ref_model.items(): # category Lj
            n_ij = len(entity_i.intersection(entity_j)) # |Ci n Lj|
            p_ij = n_ij / n_i # Precision(Ci, Lj)
            entity_purity_i = p_ij if p_ij > entity_purity_i \
                    else entity_purity_i # max_j Precision(Ci, Lj)
        model_purity += p_i * entity_purity_i

    string = 'The total purity of the loaded model compared to' + \
            ' the reference model is p =\t\t\t{:.3f}'
    return (model_purity, string)

def inverse_purity(resources, model, ref_model):
    N = len(resources)
    model_inverse_purity = 0.0
    for entity_id_i, entity_i in ref_model.items(): # category Li
        n_i  = len(entity_i)    # |Li|
        p_i = n_i / N # proportion of Li
        entity_inverse_purity_i = 0.0
        for entity_id_j, entity_j in model.items(): # cluster Cj
            n_ij = len(entity_i.intersection(entity_j)) # |Li n Cj|
            p_ij = n_ij / n_i # Precision(Li, Cj)
            entity_inverse_purity_i = p_ij if p_ij > entity_inverse_purity_i \
                    else entity_inverse_purity_i # max_j Precision(Li, Cj)
        model_inverse_purity += p_i * entity_inverse_purity_i

    string = 'The total inverse purity of the loaded model compared to' + \
            ' the reference model is inv_p =\t\t\t{:.3f}'
    return (model_inverse_purity, string)

# Van Rijsbergen's F measure, combining Purity and Inverse Purity
def F_measure(resources, model, ref_model):
    N = len(resources)
    model_F_value = 0.0
    for entity_id_i, entity_i in ref_model.items(): # category Li
        n_i = len(entity_i) # |Li|
        p_i = n_i / N # proportion of Li
        entity_F_value = 0.0
        for entity_id_j, entity_j in model.items(): # cluster Cj
            n_ij = len(entity_i.intersection(entity_j)) # |Li n Cj|
            n_j = len(entity_j) # |Cj|
            precision = n_ij / n_i # Precison(Li, Cj)
            recall = n_ij / n_j # Recall(Li, Cj)
            if (recall + precision) == 0:
                F_ij = 0
            else:
                F_ij = 2 * recall * precision / (recall + precision)
            entity_F_value = F_ij if F_ij > entity_F_value \
                    else entity_F_value # max_j F(Li, Cj)
        model_F_value += p_i * entity_F_value

    string = 'The total F measure value of the loaded model compared to' + \
            ' the reference model is F =\t\t\t{:.3f}'
    return (model_F_value, string)

# 2. Metrics based on counting pairs
def report_counting_pairs(resources, model, ref_model):
    sc_RI, str_RI = Rand_stat(resources, model, ref_model)
    sc_Jaccard, str_Jaccard = Jaccard_coef(resources, model, ref_model)
    sc_FM, str_FM = Folkes_and_Mallows(resources, model, ref_model)
    print('Scoring report of evaluation using "counting pairs":')
    print('\t', end='')
    print(str_RI.format(sc_RI))
    print('\t', end='')
    print(str_Jaccard.format(sc_Jaccard))
    print('\t', end='')
    print(str_FM.format(sc_FM))

def _similarity_matrices(resources, model, ref_model):
    # calculate the matrices
    resources = list(resources)
    N = len(resources)

    # both mat_cluster & mat_category use the same indexing

    mat_cluster = np.eye(N) # diag set to 1
    for entity_id, entity in model.items():
        entity = list(entity)
        # avoid repeatedly counting the simultaneous appearance
        for i in range(len(entity) - 1):
            u = resources.index(entity[i])
            for j in range(i + 1, len(entity)):
                v = resources.index(entity[j])
                # symmetric
                mat_cluster[u][v] = 1
                mat_cluster[v][u] = 1

    mat_category = np.eye(N) # diag set to 1
    for entity_id, entity in ref_model.items():
        entity = list(entity)
        # avoid repeatedly counting the simultaneous appearance
        for i in range(len(entity) - 1):
            u = resources.index(entity[i])
            for j in range(i + 1, len(entity)):
                v = resources.index(entity[j])
                mat_category[u][v] = 1
                mat_category[v][u] = 1

    return mat_cluster, mat_category

def _counting_pairs(resources, model, ref_model):
    N = len(resources)
    mat_cluster, mat_category = _similarity_matrices(
            resources, model, ref_model)

    # #pairs belong to same/different cluster, same/different category
    ss = 0
    sd = 0
    ds = 0
    dd = 0

    # counting pairs
    for i in range(N - 1):
        for j in range(i + 1, N): # for any two different items
            if mat_cluster[i][j] == 0 and mat_category[i][j] == 0:
                dd += 1 # different cluster, different category
            elif mat_cluster[i][j] == 0 and mat_category[i][j] == 1:
                ds += 1 # different cluster, same category
            elif mat_cluster[i][j] == 1 and mat_category[i][j] == 0:
                sd += 1 # same cluster, different category
            else:
                ss += 1 # same cluster, same category

    return ss, sd, ds, dd

def Rand_stat(resources, model, ref_model):
    ss, sd, ds, dd = _counting_pairs(resources, model, ref_model)
    string = 'The Rand statistic of the loaded model compared to' + \
            ' the reference model is R =\t\t\t{:.3f}'
    R = (ss + dd) / (ss + sd + ds + dd)
    return (R, string)

def Jaccard_coef(resources, model, ref_model):
    ss, sd, ds, dd = _counting_pairs(resources, model, ref_model)
    string = 'The Jaccard Coefficient of the loaded model compared to' + \
            ' the reference model is J =\t\t\t{:.3f}'
    J = ss / (ss + sd + ds)
    return (J, string)

def Folkes_and_Mallows(resources, model, ref_model):
    ss, sd, ds, dd = _counting_pairs(resources, model, ref_model)
    string = 'The Folkes and Mallows score of the loaded model compared to' + \
            ' the reference model is FM =\t\t\t{:.3f}'
    J = ss / (ss + sd + ds)
    FM = math.sqrt((ss / (ss + sd)) * (ss / (ss + ds)))
    return (FM, string)

# 3. Metrics based on entropy
def report_entropy_based(resources, model, ref_model):
    sc_entropy, str_entropy = entropy_measure(resources, model, ref_model)
    print('Scoring report of evaluation using "entropy-based":')
    print('\t', end='')
    print(str_entropy.format(sc_entropy))

def entropy_measure(resources, model, ref_model):
    N = len(resources)
    model_entropy = 0
    for entity_id_j, entity_j in model.items(): # cluster j
        p_j = len(entity_j) / N # proportion of cluster j
        entity_entropy_j = 0 
        for entity_id_i, entity_i in ref_model.items(): # category i
            p = len(entity_i.intersection(entity_j)) / len(entity_j)
            # Inner SUM
            if p != 0:
                entity_entropy_j += (-1) * p * math.log2(p)
            else:
                # lim -> 0
                entity_entropy_j += 0
        # Outer SUM
        model_entropy += p_j * entity_entropy_j

    string = 'The total entropy of the loaded model compared to' + \
            ' the reference model is e =\t\t\t{:.3f}'
    return (model_entropy, string)

# 4. (Original) BCubed metrics: NOT implemented
'''
def _correctness_matrix(resources, model, ref_model):
    N = len(resources)
    mat_cluster, mat_category = _similarity_matrices(
            resources, model, ref_model)

    # indexing follows mat_cluster & mat_category
    mat_correctness = np.zeros((N, N))
    for i in range(N - 1):
        for j in range(i + 1, n): # for any two different items (e, e')
            # L(e) = L(e') <==> C(e) = C(e')
            if mat_category[i][j] == 1 and mat_cluster[i][j] == 1:
                mat_correctness[i][j] = 1

    return mat_correctness
'''

# 5. Extended BCubed metrics for both disjoint/overlapping clustering
def report_BCubed_metrics(resources, model, ref_model):
    sc_bc_precision, str_bc_precision = bcubed_precision(
            resources, model, ref_model)
    sc_bc_recall, str_bc_recall = bcubed_recall(
            resources, model, ref_model)
    sc_bc_F_measure, str_bc_F_measure = bcubed_F_measure(
            resources, model, ref_model)
    print('Scoring report of evaluation using "BCubed metrics":')
    print('\t', end='')
    print(str_bc_precision.format(sc_bc_precision))
    print('\t', end='')
    print(str_bc_recall.format(sc_bc_recall))
    print('\t', end='')
    print(str_bc_F_measure.format(sc_bc_F_measure))

def _multiplicity_matrices(resources, model, ref_model):
    # calculate the matrices
    resources = sorted(list(resources))
    N = len(resources)

    # both mat_cluster_multi & mat_category_multi use the same indexing

    mat_cluster_multi = np.zeros((N, N)) # diag set to 0
    for entity_id, entity in model.items():
        entity = list(entity)
        # avoid repeatedly counting the simultaneous appearance
        for i in range(len(entity) - 1):
            u = resources.index(entity[i])
            for j in range(i + 1, len(entity)):
                v = resources.index(entity[j])
                # symmetric
                mat_cluster_multi[u][v] += 1
                mat_cluster_multi[v][u] += 1

    mat_category_multi = np.zeros((N, N)) # diag set to 0
    for entity_id, entity in ref_model.items():
        entity = list(entity)
        # avoid repeatedly counting the simultaneous appearance
        for i in range(len(entity) - 1):
            u = resources.index(entity[i])
            for j in range(i + 1, len(entity)):
                v = resources.index(entity[j])
                # symmetric
                mat_category_multi[u][v] += 1
                mat_category_multi[v][u] += 1

    return mat_cluster_multi, mat_category_multi


def _pairwise_multiplicity_precision(resources, model, ref_model):
    N = len(resources)
    mat_cluster_multi, mat_category_multi = _multiplicity_matrices(
            resources, model, ref_model)

    # indexing follows mat_cluster_multi & mat_category_multi
    mat_multi_precision = np.zeros((N, N)) # diag set to 0
    for i in range(N - 1):
        for j in range(i + 1, N): # for any two different items
            # Min(|C(e) n C(e')|, |L(e) n L(e')|) / |C(e) n C(e')|
            if mat_cluster_multi[i][j] > 0:
                # defined only when sharing at least 1 cluster
                pw_multi_precision = min(
                        mat_cluster_multi[i][j], mat_category_multi[i][j]) / \
                                mat_cluster_multi[i][j]
            else:
                # -1 otherwise
                pw_multi_precision = -1

            mat_multi_precision[i][j] = pw_multi_precision
            mat_multi_precision[j][i] = pw_multi_precision

    return mat_multi_precision


def _pairwise_multiplicity_recall(resources, model, ref_model):
    N = len(resources)
    mat_cluster_multi, mat_category_multi = _multiplicity_matrices(
            resources, model, ref_model)

    # indexing follows mat_cluster_multi & mat_category_multi
    mat_multi_recall = np.zeros((N, N)) # diag set to 0
    for i in range(N - 1):
        for j in range(i + 1, N): # for any two different items
            # Min(|C(e) n C(e')|, |L(e) n L(e')|) / |L(e) n L(e')|
            if mat_category_multi[i][j] > 0:
                # defined only when sharing at least 1 cluster
                pw_multi_recall = min(
                        mat_cluster_multi[i][j], mat_category_multi[i][j]) / \
                                mat_category_multi[i][j]
            else:
                # -1 otherwise
                pw_multi_recall = -1

            mat_multi_recall[i][j] = pw_multi_recall
            mat_multi_recall[j][i] = pw_multi_recall

    return mat_multi_recall

def bcubed_precision(resources, model, ref_model):
    N = len(resources)
    mat_multi_precision = _pairwise_multiplicity_precision(
            resources, model, ref_model)

    precision_bcubed = 0.0
    # indexing follows mat_multi_precision
    # Avg_e[ Avg_e' [Multiplicity precision(e, e')] ], for non-empty sharing
    for i in range(N - 1):
        precision_bcubed_by_row = 0.0
        cnt_valid_by_row = 0
        for j in range(i + 1, N):
            # for non-empty sharing: |C(e) n C(e')| > 0
            if mat_multi_precision[i][j] != -1:
                precision_bcubed_by_row += mat_multi_precision[i][j]
                cnt_valid_by_row += 1
        # Avg_e'
        precision_bcubed += 0 if cnt_valid_by_row == 0 else \
                precision_bcubed_by_row / cnt_valid_by_row
    # Avg_e
    precision_bcubed /= (N - 1)
    string = 'The Precision BCubed of the loaded model compared to' + \
            ' the reference model is Precision BCubed =\t\t\t{:.3f}'
    return (precision_bcubed, string)

def bcubed_recall(resources, model, ref_model):
    N = len(resources)
    mat_multi_recall = _pairwise_multiplicity_recall(
            resources, model, ref_model)

    recall_bcubed = 0.0
    # indexing follows mat_multi_recall
    # Avg_e[ Avg_e' [Multiplicity recall(e, e')] ], for non-empty sharing
    for i in range(N - 1):
        recall_bcubed_by_row = 0.0
        cnt_valid_by_row = 0
        for j in range(i + 1, N):
            # for non-empty sharing: |L(e) n L(e')| > 0
            if mat_multi_recall[i][j] != -1:
                recall_bcubed_by_row += mat_multi_recall[i][j]
                cnt_valid_by_row += 1
        # Avg_e'
        recall_bcubed += 0 if cnt_valid_by_row == 0 else \
                recall_bcubed_by_row / cnt_valid_by_row
    # Avg_e
    recall_bcubed /= (N - 1)
    string = 'The Recall BCubed of the loaded model compared to' + \
            ' the reference model is Recall BCubed =\t\t\t{:.3f}'
    return (recall_bcubed, string)

def bcubed_F_measure(resources, model, ref_model):
    bc_precision = bcubed_precision(resources, model, ref_model)[0]
    bc_recall = bcubed_recall(resources, model, ref_model)[0]
    F_score = 2 * \
            (bc_precision * bc_recall) / (bc_precision + bc_recall)
    string = 'The F-measure (BCubed) of the loaded model compared to' + \
            ' the reference model is F(BCubed) =\t\t\t{:.3f}'
    return (F_score, string)

