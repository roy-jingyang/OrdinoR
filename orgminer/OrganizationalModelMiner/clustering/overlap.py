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

def _gmm(
        profiles, n_groups,
        threshold=None, init='random', n_init=100): 
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
            none is given, then a random number (ranged [0, 1]) is used.
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
                - 'kmeans': Use the classic clustering algorithm KMeans as
                  initialization method.
                - 'plain': Simply put resource into groups by their topological
                  order in a looping fashion. Note that this is a meaniningless
                  initialization similar to a zero initialization.
            Note that if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using ' +
            'clustering-based GMM:')

    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        if init == 'mja':
            from orgminer.OrganizationalModelMiner.community.graph_partitioning import (
                    _mja)
            init_groups = _mja(profiles, n_groups)  
        elif init == 'ahc':
            from .hierarchical import _ahc
            init_groups, _ = _ahc(profiles, n_groups)
        elif init == 'kmeans':
            init_groups = list(set() for i in range(n_groups))
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n_groups, random_state=0).fit_predict(
                    profiles)
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[labels[i]].add(r)
        elif init == 'plain':
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        else:
            exit(1)
        print('Initialization done using {}:'.format(init))
    elif init == 'random':
        warm_start = False
    else:
        exit('[Error] Unrecognized parameter "{}".'.format(init))

    # step 1. Train the model
    from sklearn.mixture import GaussianMixture
    if warm_start:
        from numpy import mean
        init_means = list()
        for g in init_groups:
            init_means.append(mean(profiles.loc[list(g)].values, axis=0))
        gmm_model = GaussianMixture(
                n_components=n_groups,
                covariance_type='tied',
                tol=1e-9,
                max_iter=1000,
                n_init=1,
                random_state=0,
                means_init=init_means).fit(profiles.values)
    else:
        gmm_model = GaussianMixture(
                n_components=n_groups,
                covariance_type='tied',
                tol=1e-9,
                max_iter=1000,
                n_init=n_init,
                init_params='random').fit(profiles.values)

    # step 2. Derive the clusters as the end result
    posterior_pr = gmm_model.predict_proba(profiles.values)

    from numpy import nonzero, median, unique, count_nonzero, amin, percentile
    from collections import defaultdict
    groups = defaultdict(set)

    threshold_ub = 0
    for pr in sorted(unique(posterior_pr), reverse=True):
        mbr_mat = posterior_pr >= pr
        if count_nonzero(mbr_mat.any(axis=1)) < len(profiles):
            pass
        else:
            if count_nonzero(mbr_mat.any(axis=0)) == n_groups:
                threshold_ub = pr
                break

    if threshold_ub != 0.0:
        threshold = amin([percentile(posterior_pr, 75), threshold_ub])
        membership_total = posterior_pr >= threshold
    else:
        membership_total = posterior_pr > 0.0

    for i, membership in enumerate(membership_total):
        for j in nonzero(membership)[0]:
            groups[j].add(profiles.index[i])

    #print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

def gmm(
        profiles, n_groups,
        threshold=None, init='random', n_init=100,
        search_only=False): 
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: iterable
            The (range of) number of groups to be discovered.
        threshold: float, optional
            The threshold value for determining the resource membership. If
            none is given, then a random number (ranged [0, 1]) is used.
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
            Note taht if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
        search_only: boolean, optional
            Determine whether to search for the number of groups only or to
            perform cluster analysis based on the search result. The default is
            to perform cluster analysis after searching.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    if len(n_groups) == 1:
        return _gmm(profiles, n_groups[0], threshold, init, n_init)
    else:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import cross_validation_score
        from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
        from numpy import mean, amax
        for k in n_groups:
            # NOTE: use Cross Validation
            score = cross_validation_score(
                X=profiles, miner=_gmm,
                miner_params={
                    'n_groups': k,
                    'threshold': threshold,
                    'init': init,
                    'n_init': n_init
                },
                proximity_metric='euclidean'
            )
            '''

            # NOTE: use Silhouette score
            ogs = _gmm(profiles, k, threshold, init, n_init)
            sil_scores = silhouette_score(ogs, profiles, metric='euclidean')
            mean_sil_score = mean(list(sil_scores.values()))
            scores_clu = list()
            for g in ogs:
                if len(g) > 1:
                    score_g = mean([sil_scores[r]
                        for r in g if sil_scores[r] != 0.0])
                    max_score_g = amax([sil_scores[r]
                        for r in g if sil_scores[r] != 0.0])
                    scores_clu.append((score_g, max_score_g))
            if all([(x[1] >= mean_sil_score) for x in scores_clu]):
                # if it is a valid 'K'
                score = mean_sil_score
            else:
                # if it is an invalid 'K'
                score = float('-inf')
            '''

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _gmm(profiles, best_k, threshold, init, n_init)

def _moc(
        profiles, n_groups,
        init='random', n_init=100):
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
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
                - 'plain': Simply put resource into groups by their topological
                  order in a looping fashion. Note that this is a meaniningless
                  initialization similar to a zero initialization.
            Note that if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using ' + 
            'clustering-based MOC:')
    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        from numpy import zeros
        from pandas import DataFrame
        if init == 'mja':
            from orgminer.OrganizationalModelMiner.community.graph_partitioning import (
                    _mja)
            init_groups = _mja(profiles, n_groups)  
        elif init == 'ahc':
            from .hierarchical import _ahc
            init_groups, _ = _ahc(profiles, n_groups)
        elif init == 'kmeans':
            init_groups = list(set() for i in range(n_groups))
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n_groups, random_state=0).fit_predict(
                    profiles)
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[labels[i]].add(r)
        elif init == 'plain':
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        else:
            exit(1)
        print('Initialization done using {}:'.format(init))

        m = DataFrame(zeros((len(profiles), n_groups)), index=profiles.index)
        for i, g in enumerate(init_groups):
            for r in g:
                m.loc[r][i] = 1 # set the membership matrix as init input
    elif init == 'random':
        warm_start = False
    else:
        exit('[Error] Unrecognized parameter "{}".'.format(init))

    # step 1. Train the model
    from .classes import MOC
    if warm_start:
        moc_model = MOC(n_components=n_groups, M_init=m.values, n_init=1)
    else:
        moc_model = MOC(n_components=n_groups, n_init=n_init)

    # step 2. Derive the clusters as the end result
    mat_membership = moc_model.fit_predict(profiles.values)

    from numpy import nonzero
    from collections import defaultdict
    groups = defaultdict(set)
    for i in range(len(mat_membership)):
        # check if any valid membership exists for the resource based on
        # the results predicted by the obtained MOC model
        if mat_membership[i,:].any(): # valid if at least belongs to 1 group
            for j in nonzero(mat_membership[i,:])[0]:
                groups[j].add(profiles.index[i])
        else: # invalid (unexpected exit)
            print(mat_membership)
            exit('[Fatal error] MOC failed to produce a valid result')

    #print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

def moc(
        profiles, n_groups,
        init='random', n_init=100,
        search_only=False):
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: iterable
            The (range of) number of groups to be discovered.
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
            Note taht if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
        search_only: boolean, optional
            Determine whether to search for the number of groups only or to
            perform cluster analysis based on the search result. The default is
            to perform cluster analysis after searching.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    if len(n_groups) == 1:
        return _moc(profiles, n_groups[0], init, n_init)
    else:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import cross_validation_score
        from orgminer.Evaluation.m2m.cluster_validation import silhouette_score
        from numpy import mean, amax
        for k in n_groups:
            # NOTE: use Cross Validation
            score = cross_validation_score(
                X=profiles, miner=_moc,
                miner_params={
                    'n_groups': k,
                    'init': init,
                    'n_init': n_init
                },
                proximity_metric='euclidean'
            )
            '''

            # NOTE: use Silhouette score
            ogs, _ = _moc(profiles, k, init, n_init)
            sil_scores = silhouette_score(ogs, profiles, metric='euclidean')
            mean_sil_score = mean(list(sil_scores.values()))
            scores_clu = list()
            for g in ogs:
                if len(g) > 1:
                    score_g = mean([sil_scores[r]
                        for r in g if sil_scores[r] != 0.0])
                    max_score_g = amax([sil_scores[r]
                        for r in g if sil_scores[r] != 0.0])
                    scores_clu.append((score_g, max_score_g))
            if all([(x[1] >= mean_sil_score) for x in scores_clu]):
                # if it is a valid 'K'
                score = mean_sil_score
            else:
                # if it is an invalid 'K'
                score = float('-inf')
            '''

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _moc(profiles, best_k, init, n_init)

def _fcm(
        profiles, n_groups,
        threshold=None, init='random', n_init=100): 
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
            none is given, then a random number (ranged [0, 1]) is used.
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
                - 'plain': Simply put resource into groups by their topological
                  order in a looping fashion. Note that this is a meaniningless
                  initialization similar to a zero initialization.
            Note that if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using ' +
            'clustering-based FCM:')

    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        if init == 'mja':
            from OrganizationalModelMiner.community.graph_partitioning import (
                    _mja)
            init_groups = _mja(profiles, n_groups)  
        elif init == 'ahc':
            from .hierarchical import _ahc
            init_groups, _ = _ahc(profiles, n_groups)
        elif init == 'kmeans':
            init_groups = list(set() for i in range(n_groups))
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=n_groups, random_state=0).fit_predict(
                    profiles)
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[labels[i]].add(r)
        elif init == 'plain':
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        else:
            exit(1)
        print('Initialization done using {}:'.format(init))
    elif init == 'random':
        warm_start = False
    else:
        exit('[Error] Unrecognized parameter "{}".'.format(init))

    # step 1. Train the model
    from .classes import FCM
    if warm_start:
        from numpy import array, mean, nonzero, zeros
        init_means = list()
        for g in init_groups:
            init_means.append(mean(profiles.loc[list(g)].values, axis=0))
        fcm_model = FCM(
                n_components=n_groups,
                n_init=1,
                means_init=array(init_means))
    else:
        fcm_model = FCM(
                n_components=n_groups,
                n_init=n_init)

    # step 2. Derive the clusters as the end result
    fpp = fcm_model.fit_predict(profiles.values)

    from numpy import array, nonzero, argmax, median
    from numpy.random import choice
    from collections import defaultdict
    groups = defaultdict(set)
    for i, belonging_factor in enumerate(fpp):
        if threshold is None:
            #threshold = median(belonging_factor[belonging_factor != 0])
            threshold = 1.0 / n_groups
        membership = array([p >= threshold for p in belonging_factor])

        # check if any valid membership exists for the resource based on
        # the selection of the threshold
        if membership.any():
            for j in nonzero(membership)[0]:
                groups[j].add(profiles.index[i])
        else: # invalid, have to choose the maximum one or missing the resource
            groups[argmax(belonging_factor)].add(profiles.index[i])

    #print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

def fcm(
        profiles, n_groups,
        threshold=None, init='random', n_init=100,
        search_only=False): 
    '''
    This method is just a wrapper function of the one above, which allows a
    range of expected number of organizational groups to be specified rather
    than an exact number.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        n_groups: iterable
            The (range of) number of groups to be discovered.
        threshold: float, optional
            The threshold value for determining the resource membership. If
            none is given, then a random number (ranged [0, 1]) is used.
        init: str, optional
            The strategy used for initialization. The default is to use random
            initialization. Other options include:
                - 'mja': Use the mining method of 'Metric based on Joint 
                Activities' as initialization method.
                - 'ahc': Use the (hierarchical) method of 'Agglomerative
                  Hierarchical Clustering' as initialization method.
            Note taht if an option other than 'random' is specified, then the
            initialization is performed once only.
        n_init: int, optional
            The number of times of random initialization (if specified)
            performed before training the clustering model, ranged [1, +inf).
            The default is 100.
        search_only: boolean, optional
            Determine whether to search for the number of groups only or to
            perform cluster analysis based on the search result. The default is
            to perform cluster analysis after searching.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    if len(n_groups) == 1:
        return _fcm(profiles, n_groups[0], threshold, init, n_init)
    else:
        from orgminer.OrganizationalModelMiner.utilities import cross_validation_score
        best_k = -1
        best_score = float('-inf')
        for k in n_groups:
            score = cross_validation_score(
                X=profiles, miner=_fcm,
                miner_params={
                    'n_groups': k,
                    'threshold': threshold,
                    'init': init,
                    'n_init': n_init
                },
                proximity_metric='euclidean'
            )
            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _fcm(profiles, best_k, threshold, init, n_init)

