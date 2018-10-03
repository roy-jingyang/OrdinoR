# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining overlapping orga-
nizational models, based on the use of clustering techniques. These methods are
"profile-based", meaning that resource profiles should be used as input.

Methods include:
    1. GMM (Gaussian Mixture Model) (J.Yang et al.)
    2. MOC (Model-based Overlapping Clusttering) (J.Yang et al.)
    3. FCM (Fuzzy c-Means)
'''

def gmm(
        profiles, n_groups,
        threshold=None, cov_type='spherical', warm_start_input_fn=None): 
    '''
    This method implements the algorithm of using Gaussian Mixture Model
    for mining an overlapping organizational model from the given event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        threshold: float, optional
            The threshold value for determining the resource membership. If
            none is given, then the algorithm produces a disjoint partition as
            result.
        cov_type: str, optional
            String describing the type of covariance parameters to use. The
            default is 'spherical'.For detailed explanation, see
            http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        warm_start_input_fn: str, optional
            Filename of the initial guess of clustering.
            The default is None, meaning warm start is NOT used.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''

    from collections import defaultdict

    print('Applying overlapping organizational model mining using ' +
            'clustering-based GMM:')
    # step 1. Importing warm-start (initial guess of clustering) from file
    gmm_warm_start = (warm_start_input_fn is not None)
    if gmm_warm_start:
        from csv import reader
        with open(warm_start_input_fn, 'r') as f:
            is_header = True
            init_groups = defaultdict(lambda: set())
            count_groups = 0
            for row in reader(f):
                if is_header:
                    is_header = False
                else:
                    group_id = row[0]
                    for r in row[-1].split(';'):
                        init_groups[group_id].add(r)
                    count_groups += 1

        if n_groups != count_groups:
            print('[Warning] Inequal group size {} != {}.'.format(
                n_groups, count_groups))
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
                n_init=500,
                init_params='random').fit(profiles.values)

    # step a3. Deriving the clusters as the end result
    posterior_pr = gmm_model.predict_proba(profiles)
    from numpy import array, amax, nonzero, argmax
    from numpy.random import choice
    groups = defaultdict(lambda: set())
    # TODO: more pythonic way required
    for i in range(len(posterior_pr)):
        resource_postpr = posterior_pr[i]
        if threshold is None:
            # TODO: is_disjoint option
            #threshold = amax(resource_postpr)
            # TODO: randomly select a threshold
            threshold = choice(resource_postpr[resource_postpr != 0])
        membership = array([p >= threshold for p in resource_postpr])

        # check if any valid membership exists for the resource based on
        # the selection of the threshold
        if membership.any():
            for j in nonzero(membership)[0]:
                groups[j].add(profiles.index[i])
        else: # invalid, have to choose the maximum one or missing the resource
            groups[argmax(resource_postpr)].add(profiles.index[i])

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

def moc(
        profiles, n_groups,
        warm_start_input_fn=None):
    '''
    This method implements the algorithm of using Model-based Overlapping
    Clustering for mining an overlapping organizational model from the given
    event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        warm_start_input_fn: str, optional
            Filename of the initial guess of clustering.
            The default is None, meaning warm start is NOT used.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''

    print('Applying overlapping organizational model mining using ' + 
            'clustering-based MOC:')
    # step 1. Importing warm-start (initial guess of clustering from file
    moc_warm_start = (warm_start_input_fn is not None)
    if moc_warm_start:
        from numpy import zeros
        from pandas import DataFrame
        m = DataFrame(zeros((len(profiles), n_groups)), index=profiles.index)
        from csv import reader
        with open(warm_start_input_fn, 'r') as f:
            is_header = True
            count_groups = 0
            for row in reader(f):
                if is_header:
                    is_header = False
                else:
                    for r in row[-1].split(';'):
                        m.loc[r][count_groups] = 1 # equals to m[i,j]
                    count_groups += 1
        if n_groups != count_groups:
            print('[Warning] Inequal group size {} != {}.'.format(
                n_groups, count_groups))
        else:
            print('Initial guess imported from file "{}".'.format(
                warm_start_input_fn))

    # step 2. Training the model
    from .classes import MOC
    if moc_warm_start:
        moc_model = MOC(n_components=n_groups, M_init=m.values)
    else:
        # TODO: is_disjoint option
        #moc_model = MOC(n_components=n_groups, n_init=500, is_disjoint=True)
        moc_model = MOC(n_components=n_groups, n_init=500)

    mat_membership = moc_model.fit_predict(profiles.values)

    # step 3. Deriving the clusters as the end result
    from numpy import nonzero
    from collections import defaultdict
    groups = defaultdict(lambda: set())
    # TODO: more pythonic way required
    for i in range(len(mat_membership)):
        # check if any valid membership exists for the resource based on
        # the results predicted by the obtained MOC model
        if mat_membership[i,:].any(): # valid if at least belongs to 1 group
            for j in nonzero(mat_membership[i,:])[0]:
                groups[j].add(profiles.index[i])
        else: # invalid (unexpected exit)
            exit('[Fatal error] MOC failed to produce a valid result')

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

# TODO
def fcm(
        profiles, n_groups,
        threshold=None, warm_start_input_fn=None):
    '''
    This method implements the algorithm of using Fuzzy c-Means for mining an
    overlapping organizational model from the given event log.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: int
            The number of groups to be discovered.
        threshold: float, optional
            The threshold value for determining the resource membership. If
            none is given, then the algorithm produces a disjoint partition as
            result.
        warm_start_input_fn: str, optional
            Filename of the initial guess of clustering. The file should be
            formatted as:
                Group ID, resource; resource; ...
            with each line in the CSV file representing a group.
            The default is None, meaning warm start is NOT used.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using ' +
            'clustering-based FCM:')

    from collections import defaultdict

    # step 1. Importing warm-start (initial guess of clustering) from file
    fcm_warm_start = (warm_start_input_fn is not None)
    if fcm_warm_start:
        from csv import reader
        with open(warm_start_input_fn, 'r') as f:
            is_header = True
            init_groups = defaultdict(lambda: set())
            count_groups = 0
            for row in reader(f):
                if is_header:
                    is_header = False
                else:
                    group_id = row[0]
                    for r in row[-1].split(';'):
                        init_groups[group_id].add(r)
                    count_groups += 1

        if n_groups != count_groups:
            print('[Warning] Inequal group size {} != {}.'.format(
                n_groups, count_groups))
        else:
            print('Initial guess imported from file "{}".'.format(
                warm_start_input_fn))

    # step 2. Training the model and obtain the results
    from .classes import FCM
    if fcm_warm_start:
        from numpy import array, mean, nonzero
        init_means = list()
        for k in sorted(init_groups.keys()):
            init_means.append(mean(
                profiles.loc[list(init_groups[k])].values, axis=0))
        fcm_model = FCM(n_components=n_groups, means_init=array(init_means),
                threshold='random')
    else:
        fcm_model = FCM(n_components=n_groups, n_init=500, threshold='random')

    fpp = fcm_model.fit_predict(profiles.values)
    
    # step 3. Deriving the clusters as the end result
    from numpy import array, amax, nonzero, argmax
    from numpy.random import choice
    groups = defaultdict(lambda: set())
    for i in range(len(fpp)):
        # check if any valid membership exists for the resource based on
        # the selection of the threshold
        membership = fpp[i,:]
        if membership.any():
            for j in nonzero(membership)[0]:
                groups[j].add(profiles.index[i])
        else: # invalid, have to choose the maximum one or missing the resource
            groups[argmax(resource_wt)].add(profiles.index[i])

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

