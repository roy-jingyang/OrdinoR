#! /usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This module contains the implementation of the method of mining overlapping
orgnizational models using Gaussian Mixture Model, proposed by Yang et al.
(ref. J.Yang et al., BPM 2018).
'''

def mine(profiles,
        n_groups,
        cov_type='spherical',
        warm_start_input_fn=None): 
    '''
    This method implements the algorithm of using Gaussian Mixture Model
    for mining an overlapping organizational model from the given event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        cov_type: str, optional
            String describing the type of covariance parameters to use. See
            http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
            for details.
        warm_start_input_fn: str, optional
            Filename of the initial guess of clustering. The file should be
            formatted as:
                Group ID, resource; resource; ...
            with each line in the CSV file representing a group.
            The default is None, meaning warm start is NOT used.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''
    from collections import defaultdict

    print('Applying overlapping organizational model mining using ' +
            'clustering-based GMM:')
    # step 1. Importing warm-start (initial guess of clustering) from file
    gmm_warm_start = (warm_start_input_fn is not None)
    if gmm_warm_start:
        from csv import reader
        with open(warm_start_input_fn, 'r') as f:
            init_groups = defaultdict(lambda: set())
            for row in reader(f):
                group_id = row[0]
                for r in row[1].split(';'):
                    init_groups[group_id].add(r)
        if n_groups != len(init_groups):
            exit('Invalid initial guess detected. Exit with error.')
        else:
            print('Initial guess imported from file "{}".'.format(
                warm_start_input_fn))

    # step a2. Training the model
    from sklearn.mixture import GaussianMixture
    if gmm_warm_start:
        init_wt = [1.0 / n_groups] * n_groups
        from numpy import mean
        init_means = list()
        for k in sorted(init_groups.keys()):
            init_means.append(mean(
                profiles.loc[list(init_groups[k])].values, axis=0))
        gmm_model = GaussianMixture(
                n_components=n_groups,
                covariance_type=cov_type,
                n_init=1,
                weights_init=init_wt,
                means_init=init_means).fit(profiles.values)
    else:
        gmm_model = GaussianMixture(
                n_components=n_groups,
                covariance_type=cov_type,
                n_init=100,
                init_params='random').fit(profiles.values)

    # step a3. Setting thresholds to determine membership for each data point
    posterior_pr = gmm_model.predict_proba(profiles)
    print('Input a threshold value [0, 1), in order to determine the ' +
            'resource membership (Enter to choose the max., ' + 
            'creating disjoint groups:', end=' ')
    from numpy import amax
    user_selected_threshold = input()
    user_selected_threshold = float(user_selected_threshold) if \
            user_selected_threshold != '' else None

    # step a4. Deriving the clusters as the end result
    from numpy import nonzero, argmax
    og = defaultdict(lambda: set())
    # TODO: more pythonic way required
    for i in range(len(posterior_pr)):
        resource_postpr = posterior_pr[i]
        if not user_selected_threshold:
            user_selected_threshold = amax(resource_postpr)
        membership = [p >= user_selected_threshold for p in resource_postpr]
        # check if any valid membership exists for the resource based on
        # the selection of the threshold
        if len(nonzero(membership)[0]) > 0: # valid
            for j in nonzero(membership)[0]:
                og[j].add(profiles.index[i])
        else: # invalid, have to choose the maximum one or missing the resource
            og[argmax(resource_postpr)].add(profiles.index[i])

    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

