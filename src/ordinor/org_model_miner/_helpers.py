from copy import copy
from itertools import product
from functools import partial
import multiprocessing
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cdist

def cross_validation_score(
    X, 
    miner, 
    miner_params,
    proximity_metric='euclidean', 
    cv_fold=0.2):
    """
    Cross validation method for determining an appropriate number of
    clusters for group discovery techniques based on clustering-liked
    algorithms.

    Parameters
    ----------
    X : pandas.DataFrame
        Resource profiles as input to an organizational model miner.
    miner : function
        An organizational model miner.
    miner_params : dict
        Other keyword parameters for the specified miner.
    proximity_metric : str, optional, default 'euclidean'
        Metric for measuring the distance while calculating proximity. 
        This should remain consistent with that employed by the specific 
        mining method. Defaults to ``'euclidean'``, meaning that
        euclidean distance is used for measuring proximity.
    cv_fold : int, or float in range (0, 1.0), default 0.2
        The number of folds to be used for cross validation. 
        If an integer is given, then it is used as the fold number; 
        if a float is given, then a corresponding percentage of data 
        will be used as the test fold. Defaults to ``0.2``, meaning that
        20% of data will be used for testing while the other 80% used for
        training.

    Returns
    -------
    float
        The validation score as a result.

    Notes
    -----
    Refer to scipy.spatial.distance.pdist for more detailed explanation 
    of proximity metrics.
    """
    scores = list()
    # split input dataset into specific number of folds
    index = copy(list(X.index))
    if type(cv_fold) is float and 0 < cv_fold and cv_fold < 1.0:
        cv_fold = int(1.0 / cv_fold)
    index_folds = np.array_split(index, cv_fold)    

    print('Using cross validation with {} folds:'.format(cv_fold))

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

        # calculate the cluster centroid
        # NOTE: different definitions of centroids may be used
        cluster_centroids = list()
        for c in clusters:
            cluster_centroids.append(
                np.mean([X.loc[ix] for ix in c], axis=0)
            )
        cluster_centroids = np.array(cluster_centroids) 

        # evaluate using the test set
        sum_closest_proximity = 0.0
        for ix in test_set_index:
            x = X.loc[ix].values.reshape((1, len(X.loc[ix])))
            sum_closest_proximity += np.amin(
                cdist(x, cluster_centroids, metric=proximity_metric)
            )
        scores.append((-1) * sum_closest_proximity)

    return np.mean(scores)


def grid_search(func_core, params_config, func_eval_score):
    """
    This method provides grid search functionality with multiprocessing.
    
    For any core function and its parameter field along with range of 
    values under test, a grid search will be performed to select the
    parameter configuration that leads to the best (highest-scored)
    solution, evaluated by a user-provided function. 

    Parameters
    ----------
    func_core : function
        Core function under grid search. The provided function must
        return a value (or values) that can be accepted as input
        parameters for `func_eval_score`.

    params_config : dict of lists
        A Python dictionary that specifies the range of grid search. Each
        of the key(s) correspond to a parameter from `func_core`, for
        which the value defines a range of candidate values to by used
        for search.

    func_eval_score : function
        User-provided function for evaluating an instance during the
        search process. The provided function must take as input the
        return from `func_core` and calculates a score, for which a
        higher value indicates a better solution, and vice versa.

    Returns
    -------
    solution : (return type depending on `func_core`)
        The best (highest-scored) solution returned from `func_core`.

    params_best : dict
        The parameter settings associated with the best (highest-scored)
        solution, encoded in a Python dictionary.
    """
    l_tuples_all_configs = list()
    for param_field, param_value_range in params_config.items():
        l_tuples_configs = list()
        for value in param_value_range:
            l_tuples_configs.append((param_field, value)) 
        l_tuples_all_configs.append(l_tuples_configs)

    l_dicts_all_configs = list()
    for param_config in product(*l_tuples_all_configs):
        params = dict()
        for (field, value) in param_config:
            params[field] = value
        l_dicts_all_configs.append(params)

    with multiprocessing.Pool() as p:
        params_best = max(
            p.map(partial(
                    _grid_search_wrapper, 
                    func_core,
                    func_eval_score
                ), 
                l_dicts_all_configs
            ),
            key=itemgetter(1))[0]

    solution = func_core(**params_best)
    return solution, params_best


def _grid_search_wrapper(func_core, func_eval_score, params):
    func_core_ret = func_core(**params)
    score = func_eval_score(func_core_ret)
    return params, score
