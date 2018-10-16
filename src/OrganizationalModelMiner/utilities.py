# -*- coding: utf-8 -*-

'''
This module provides the necessary helper functions for organizational model
mining methods.
'''

def cross_validation_score(
        X, miner, miner_params,
        proximity_metric='euclidean', cv_fold=0.25):
    '''
    This method implements the cross validation strategy for determining an
    appropriate number of clusters ('K') for organizational model miners that
    employ a clustering-liked algorithm.

    Params:
        X: DataFrame
            The input feature vectors for the miner method.
        miner: function
            The organizational model mining method to be validated.
        miner_params: dict
            The parameters for the mining method.
        proximity_metric: str
            Choice of metrics for measuring the distance while calculating the
            proximity. Refer to scipy.spatial.distance.pdist for more detailed
            explanation. This should be consistent with that employed within
            the mining method.
        cv_fold: int, or float in (0, 1)
            The number of folds to be used for cross validation. If an integer
            K is given, then K is used as the fold number; if a floating number
            P is given, then (P * 100)% data will be used as the test fold,
            which means that the fold number will be approximately (1 / P).
    Returns:
        score: float
            The result validation score.
    '''
    scores = list()

    # split input dataset into specific number of folds
    from copy import copy
    index = copy(list(X.index))
    from numpy.random import shuffle
    shuffle(index)
    from numpy import array_split
    if type(cv_fold) is float and 0 < cv_fold and cv_fold < 1.0:
        cv_fold = int(1.0 / cv_fold)
    index_folds = array_split(index, cv_fold)    

    print('Using cross validation with {} folds:'.format(cv_fold))

    from numpy import array, mean # TODO: different definition of centroids
    from scipy.spatial.distance import cdist
    for i in range(cv_fold):
        # split test set and train set
        test_set_index = index_folds[i]
        train_set_index = list()
        for j in range(cv_fold):
            if j != i:
                train_set_index.extend(index_folds[j])
       
        # find the clusters using the train set
        result = miner(X.loc[train_set_index], **miner_params)
        clusters = result[0] if type(result) is tuple else result

        # calculate the cluster centroids
        cluster_centroids = list()
        for c in clusters:
            cluster_centroids.append(mean([X.loc[ix] for ix in c], axis=0))
        cluster_centroids = array(cluster_centroids) 

        # evaluate using the test set
        sum_closest_proximity = 0.0
        for ix in test_set_index:
            x = X.loc[ix].values.reshape((1, len(X.loc[ix])))
            sum_closest_proximity += min(
                    cdist(x, cluster_centroids, metric=proximity_metric)[0])
        scores.append((-1) * sum_closest_proximity)

    return mean(scores)

