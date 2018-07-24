#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of the method of mining overlapping
orgnizational models using Gaussian Mixture Model, proposed by Yang et al.
(ref. J.Yang et al., BPM 2018).
'''

def mine(profiles,
        n_groups=None): 
    '''
    This function implements the main algorithm of using Gaussian Mixture Model
    for mining an overlapping organizational model from the given event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int, optional
            The number of groups to be discovered. The default is 'None', which
            implies that user should select the appropriate number of groups by
            using the visualizing tools provided.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''

import numpy as np
import copy
#import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.mixture import GaussianMixture
#from EvaluationOptions import Unsupervised

#def mine(cases):
def mine(cases, num_c):
    print('Applying Gaussian Mixture Model:')
    '''
    if threshold < 0 or threshold >= 1:
        print('Threshold for converting to determined clustering must value' +\
                ' between [0, 1) !')
        print('Warning: Disjoint clustering option unavailable!')
        exit(1)
    '''

    # constructing the performer-activity matrix from event logs
    counting_mat = defaultdict(lambda: defaultdict(lambda: 0))
    activity_index = []

    for caseid, trace in cases.items():
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][1]
            if activity in activity_index:
                pass
            else:
                activity_index.append(activity)
            counting_mat[resource][activity] += 1

    resource_index = list(counting_mat.keys())
    profile_mat = np.zeros((len(resource_index), len(activity_index)))
    for i in range(len(resource_index)):
        activities = counting_mat[resource_index[i]]
        for act, count in activities.items():
            j = activity_index.index(act)
            profile_mat[i][j] = count

    # logrithm preprocessing (van der Aalst, 2005)
    profile_mat = np.log(profile_mat + 1)

    # Use GMM for probabilistic cluster assignments
    # Searching for an appropriate model among settings of:
    # n_components, covariance_type; init_params (manually adjust)
    # Note: Fit only, not ready for predicting yet
    # search settings (1): k_cluster
    #k_cluster_MIN = 2
    #k_cluster_MAX = len(resource_index) - 1
    k_cluster_MIN = k_cluster_MAX = int(num_c)
    k_cluster_step = 1
    # search settings (2): covariance_type (default: full)
    #cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['spherical']

    # Read in initialized membership if warm start
    #print('Filename for warm-start input (enter if none):', end=' ')
    #f_membership = input()
    f_membership = ''
    dict_membership = defaultdict(lambda: set())
    if f_membership is not '':
        with open(f_membership, 'r') as f:
            for line in f:
                line = line.strip()
                i = resource_index.index(line.split(',')[0])
                j = int(line.split(',')[1])
                dict_membership[j].add(i)
        gmm_warm_start = True
    else:
        gmm_warm_start = False

    gmm_models = list()
    gmm_models_flattened = list()
    for cv_type in cv_types: # search 1
        k_cluster = k_cluster_MIN
        models_by_cv_type = list()
        while k_cluster <= k_cluster_MAX: # search 2
            if gmm_warm_start and k_cluster == len(dict_membership):
                print('warm-start allowed')
                weights = [1.0 / k_cluster] * k_cluster
                means = list()
                for k in sorted(dict_membership.keys()):
                    xs = dict_membership[k]
                    means.append(np.mean(
                        [profile_mat[x] for x in xs], axis=0))
                gmm = GaussianMixture(
                        random_state=0,
                        weights_init=weights,
                        means_init=means,
                        n_components=k_cluster,
                        covariance_type=cv_type).fit(profile_mat)
                        #init_params='kmeans').fit(profile_mat)
            else:
                gmm = GaussianMixture(
                        n_components=k_cluster,
                        n_init=100,
                        covariance_type=cv_type,
                        init_params='random').fit(profile_mat)
                        #init_params='kmeans').fit(profile_mat)

            bic_score = gmm.bic(profile_mat)
            models_by_cv_type.append((gmm, bic_score))
            gmm_models_flattened.append((gmm, bic_score))
            k_cluster += k_cluster_step
        gmm_models.append(models_by_cv_type)

    '''
    # Visualizing the results (1): Select an appropriate model (cv_type)
    #f, axarr = plt.subplots(2, sharex=True)
    plt.figure(0)
    #color_types = ['blue', 'green', 'red', 'orange']
    color_types = ['blue']
    plt.xlabel('#clusters')
    x = np.arange(k_cluster_MIN, k_cluster_MAX + k_cluster_step) # #clusters
    plt.xticks(x)

    # bar chart for BIC score
    plt.ylabel('Score - BIC')
    y = list()
    for i, (cv_type, color_type) in enumerate(zip(cv_types, color_types)):
        xpos = x + 0.2 * (i - 2)
        y.append(plt.bar(
            xpos, np.array([m[1] for m in gmm_models[i]]),
            width=0.2, color=color_type))
        lowest = np.min([m[1] for m in gmm_models[i]])
        average = np.mean([m[1] for m in gmm_models[i]])
        argmin = np.argmin([m[1] for m in gmm_models[i]])
        plt.plot(x[argmin], 0,
                color=color_type, marker='x', markersize=15, markeredgewidth=2)
        print('{}: Suggested #cluster = {} with BIC = {}; Avg. = {}'.format(
            cv_type, x[argmin], lowest, average))
    plt.legend([b[0] for b in y], cv_types)
    #plt.show()
    '''
    
    '''
    print('Top-5 models with lower BIC suggested:')
    gmm_models_flattened.sort(key=lambda m: m[1])
    for i in range(5):
        print('\tcv_type = {}, #cluster = {}, BIC = {}'.format(
            gmm_models_flattened[i][0].covariance_type,
            gmm_models_flattened[i][0].n_components,
            gmm_models_flattened[i][1]))
    '''

    '''
    # select a covariance type
    #print('Select an appropriate model: covariance_type')
    #choice_cv_type = str(input())
    choice_cv_type = 'spherical'

    # Visualizing the results (2): Select an appropriate model (#clusters)
    plt.figure(1)
    plt.xlabel('#clusters')
    x = np.arange(k_cluster_MIN, k_cluster_MAX + k_cluster_step) # #clusters
    plt.xticks(x)

    # line chart for likelihood score (goal of maximization)
    plt.ylabel('Score - likelihood')
    i = cv_types.index(choice_cv_type)
    #TODO: try to add in var(k) score instead of likelihood
    plt.plot(x, np.array([m[0].score(profile_mat) for m in gmm_models[i]]),
        color=color_types[i], linestyle='dashed', marker='*',
        label=choice_cv_type)
    plt.legend()
    #plt.show()

    # select #clusters
    print('Select an appropriate model: #components')
    choice_k_cluster = int(input())
    i = list(range(k_cluster_MIN, k_cluster_MAX + k_cluster_step,
        k_cluster_step)).index(choice_k_cluster)
    choice_gmm = gmm_models[cv_types.index(choice_cv_type)][i][0]
    print(choice_gmm.n_components)

    '''
    # search settings (3): cut threshold
    choice_gmm = gmm_models[0][0][0]
    resp = choice_gmm.predict_proba(profile_mat)
    '''
    # TODO: these lines help select the cut_threshold
    print('For the current fit: #cluster = {}'.format(choice_k_cluster),
            end='\n\t')
    print('------Include 0s------', end='\n\t')
    print('MIN(Pr) = {}'.format(np.min(resp)), end=', ')
    print('MAX(Pr) = {}'.format(np.max(resp)), end=', ')
    print('Mean(Pr) = {}'.format(np.mean(resp)), end=', ')
    print('Median(Pr) = {}'.format(np.median(resp)))
    print('\n\t------Exclude 0s------', end='\n\t')
    print('{} of non-0s'.format(np.count_nonzero(resp)), end='\n\t')
    print('MIN(Pr) = {}'.format(np.min(resp[resp > 0])), end=', ')
    print('MAX(Pr) = {}'.format(np.max(resp[resp > 0])), end=', ')
    print('Mean(Pr) = {}'.format(np.mean(resp[resp > 0])), end=', ')
    print('Median(Pr) = {}'.format(np.median(resp[resp > 0])))
    '''

    # Visualizing the results (3): Select a threshold lambda for cutting
    # TODO: choice of threshold
    print('Input a threshold value [0, 1) to determine the resource membership:', end=' ')
    threshold = float(input())
    #determined = resp > threshold
    determined = list()
    for row in resp:
        #threshold = np.random.choice(row)

        # Constrained
        # WABO: round = ceil = 4
        # BPIC Open: round = 1, ceil = 2
        # BPIC Closed: round = 1, ceil = 2
        #threshold = np.random.choice(sorted(row, reverse=True)[:2])
        #threshold = sorted(row, reverse=True)[2-1]

        #threshold = np.amax(row) # disjoint

        determined.append([(x >= threshold) for x in row])
    determined = np.array(determined)
    # shape: (n_samples,)
    labels = list()
    for row in determined:
        labels.append(tuple(l for l in np.nonzero(row)[0]))
    labels = np.array(labels)

    # evaluation (unsupervised) goes here
    # only account for valid solution
    is_overlapped = type(labels[0]) != np.int64
    if is_overlapped or len(np.unique(labels)) > 1:
        print('Warning: ONLY VALID solutions are accounted.')
        #total_within_cluster_var = Unsupervised.within_cluster_variance(
        #        profile_mat, labels, is_overlapped)
        #silhouette_score = Unsupervised.silhouette_score(
        #        profile_mat, labels, is_overlapped)
        total_within_cluster_var = -1
        silhouette_score = -1
        solution = (labels, total_within_cluster_var, silhouette_score)

        #print('score: var(k) = {:.3f}'.format(solution[1]))
        #print('score: silhouette = {:.3f}'.format(solution[2]))

        entities = defaultdict(lambda: set())
        for i in range(len(solution[0])):
            if is_overlapped: # overlapped
                for l in solution[0][i]:
                    entity_id = l
                    entities[l].add(resource_index[i])
            else:
                # only 1 cluster assigned for each
                entity_id = int(solution[0])
                entities[entity_id].add(resource_index[i])

        print('Clustering results are ' + \
                ('overlapped.' if is_overlapped else 'disjoint.'))
        print('{} organizational entities extracted.'.format(len(entities)))
        return copy.deepcopy(entities)
    else:
        print('Unexpected solution: is_overlapped = {}'.format(is_overlapped))
        exit()

