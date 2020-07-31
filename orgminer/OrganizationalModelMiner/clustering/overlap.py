# -*- coding: utf-8 -*-

"""This module contains the implementation of overlapping organizational 
mining methods, based on the use of clustering techniques.
"""
def _gmm(profiles, n_groups, threshold, init='random', n_init=100): 
    """Apply the classic Gaussian Mixture Model (GMM) to discover 
    resource groups [1]_.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    threshold : float
        A given threshold value in range [0, 1.0] for producing
        determined clustering from the fuzzy clustering results from GMM.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.

    Returns
    -------
    list of frozensets
        Discovered resource groups.
    
    Raises
    ------
    ValueError
        If the specified option for initialization is invalid.
    
    See Also
    --------
    sklearn.mixture.GaussianMixture
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    print('Applying overlapping clustering-based GMM:')
    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        if init == 'mja':
            from orgminer.OrganizationalModelMiner.community.graph_partitioning\
                import _mja
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
        else: # init == 'plain'
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        print('Initialization done using {}:'.format(init))
    elif init == 'random':
        warm_start = False
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'init', init))

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
    '''
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
    '''
    membership_total = posterior_pr >= threshold

    from collections import defaultdict
    groups = defaultdict(set)
    for i, membership in enumerate(membership_total):
        for j in nonzero(membership)[0]:
            groups[j].add(profiles.index[i])

    return [frozenset(g) for g in groups.values()]


def gmm(profiles, n_groups, threshold, init='random', n_init=100,
    search_only=False): 
    """Apply the classic Gaussian Mixture Model (GMM) to discover 
    resource groups [1]_.

    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    threshold : float
        A given threshold value in range [0, 1.0] for producing
        determined clustering from the fuzzy clustering results from GMM.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after search searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    sklearn.mixture.GaussianMixture
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    if type(n_groups) is int:
        return _gmm(profiles, n_groups, threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _gmm(profiles, n_groups[0], threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import \
            cross_validation_score
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

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _gmm(profiles, best_k, threshold, init, n_init)
    else:
        raise TypeError('Invalid type for parameter ``{}``: {}'.format(
            'n_groups', type(n_groups)))


def _moc(profiles, n_groups, init='random', n_init=100):
    """Apply the Model-based Overlapping Clustering (MOC) to discover 
    resource groups [1]_.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.

    Returns
    -------
    list of frozensets
        Discovered resource groups.

    Raises
    ------
    ValueError
        If the specified option for initialization is invalid.
    RuntimeError
        If no valid result could be produced.

    See Also
    --------
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    print('Applying overlapping clustering-based MOC:')
    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        from numpy import zeros
        from pandas import DataFrame
        if init == 'mja':
            from orgminer.OrganizationalModelMiner.community.graph_partitioning\
                import _mja
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
        else: # init == 'plain'
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        print('Initialization done using {}:'.format(init))

        m = DataFrame(zeros((len(profiles), n_groups)), index=profiles.index)
        for i, g in enumerate(init_groups):
            for r in g:
                m.loc[r][i] = 1 # set the membership matrix as init input
    elif init == 'random':
        warm_start = False
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'init', init))

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
            raise RuntimeError('No valid result could be produced.')

    return [frozenset(g) for g in groups.values()]


def moc(profiles, n_groups, init='random', n_init=100,
    search_only=False):
    """Apply the Model-based Overlapping Clustering (MOC) to discover 
    resource groups [1]_.

    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Yang, J., Ouyang, C., Pan, M., Yu, Y., & ter Hofstede, A.
       (2018). Finding the Liberos: Discover Organizational Models with
       Overlaps. In *Proceedings of the 16th International Conference on
       Business Process Management*, pp. 339-355. Springer, Cham.
       `<https://doi.org/10.1007/978-3-319-98648-7_20>`_
    """
    if type(n_groups) is int:
        return _moc(profiles, n_groups, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _moc(profiles, n_groups[0], init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        best_k = -1
        best_score = float('-inf')
        from orgminer.OrganizationalModelMiner.utilities import \
            cross_validation_score
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

            if score > best_score:
                best_score = score
                best_k = k

        print('-' * 80)
        print('Selected "K" = {}'.format(best_k))
        if search_only:
            return best_k
        else:
            return _moc(profiles, best_k, init, n_init)
    else:
        raise TypeError('Invalid type for parameter `{}`: {}'.format(
            'n_groups', type(n_groups)))


def _fcm(profiles, n_groups, threshold, init='random', n_init=100): 
    """Apply the Fuzzy C-Means (FCM) [1]_ to discover resource groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    threshold : float
        A given threshold value in range [0, 1.0] for producing
        determined clustering from the fuzzy clustering results from FCM.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.

    Returns
    -------
    list of frozensets
        Discovered resource groups.

    Raises
    ------
    ValueError
        If the specified option for initialization is invalid.

    See Also
    --------
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Tan, P. N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).
       *Introduction to Data Mining*.
    """
    print('Applying overlapping clustering-based FCM:')
    # step 0. Perform specific initialization method (if given)
    if init in ['mja', 'ahc', 'kmeans', 'plain']:
        warm_start = True
        if init == 'mja':
            from orgminer.OrganizationalModelMiner.community.graph_partitioning\
                import _mja
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
        else: # init == 'plain'
            init_groups = list(set() for i in range(n_groups))
            for i, r in enumerate(sorted(profiles.index)):
                init_groups[i % n_groups].add(r)
        print('Initialization done using {}:'.format(init))
    elif init == 'random':
        warm_start = False
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'init', init))

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

    return [frozenset(g) for g in groups.values()]


def fcm(profiles, n_groups, threshold, init='random', n_init=100,
    search_only=False): 
    """Apply the Fuzzy C-Means (FCM) [1]_ to discover resource groups.

    This method allows a range of expected number of organizational
    groups to be specified rather than an exact number. It may also act 
    as a helper function for determining a proper selection of number of 
    groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int, or list of ints
        Expected number of resource groups, or a list of candidate
        numbers to be determined.
    threshold : float
        A given threshold value in range [0, 1.0] for producing
        determined clustering from the fuzzy clustering results from FCM.
    init : {'random', 'mja', 'ahc', 'kmeans', 'plain'}, optional, \
        default 'random'
        Options for deciding the strategy for initialization. Could be 
        one of the following:

            - ``'random'``, use random initialization.
            - ``'mja'``, use the mining method Metric based on Joint 
              Activities as initialization method.
            - ``'ahc'``, use the hierarchical method of Agglomerative
              Hierarchical Clustering as initialization method.
            - ``'kmeans'``, use the classic clustering algorithm kMeans 
              as initialization method.
            - ``'plain'``, simply put resources into groups by in the 
              order of their ids.

        Note that if an option other than ``'random'`` is specified, then
        the initialization is performed once only.
    n_init : int, optional, default 100
        Number of times of random initialization (if specified) performs
        before training the clustering model.
    search_only : bool, optional, default False
        A boolean flag indicating whether to search for the number of
        groups only or to perform group discovery based on the search
        result. Defaults to ``False``, i.e., to perform group discovery
        after searching.

    Returns
    -------
    best_k : int
        The suggested selection of number of groups (if `search_only` is
        True).
    list of frozensets
        Discovered resource groups (if `search_only` is False).

    Raises
    ------
    TypeError
        If the parameter type for `n_groups` is unexpected.

    See Also
    --------
    orgminer.OrganizationalModelMiner.community.graph_partitioning.mja
    orgminer.OrganizationalModelMiner.clustering.hierarchical.ahc
    sklearn.cluster.KMeans

    References
    ----------
    .. [1] Tan, P. N., Steinbach, M., Karpatne, A., & Kumar, V. (2018).
       *Introduction to Data Mining*.
    """
    if type(n_groups) is int:
        return _fcm(profiles, n_groups, threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) == 1:
        return _fcm(profiles, n_groups[0], threshold, init, n_init)
    elif type(n_groups) is list and len(n_groups) > 1:
        from orgminer.OrganizationalModelMiner.utilities import \
            cross_validation_score
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
    else:
        raise TypeError('Invalid type for parameter `{}`: {}'.format(
            'n_groups', type(n_groups)))

