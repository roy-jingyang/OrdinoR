# -*- coding: utf-8 -*-

"""This module provides the necessary helper functions for 
organizational model mining methods.
"""
from deprecated import deprecated

def cross_validation_score(
    X, miner, miner_params,
    proximity_metric='euclidean', cv_fold=0.2):
    """This method implements the cross validation method for 
    determining an appropriate number of clusters for those 
    organizational model miners employing clustering-liked algorithms.

    Parameters
    ----------
    X : DataFrame
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
    from copy import copy
    index = copy(list(X.index))
    from numpy import array_split
    if type(cv_fold) is float and 0 < cv_fold and cv_fold < 1.0:
        cv_fold = int(1.0 / cv_fold)
    index_folds = array_split(index, cv_fold)    

    print('Using cross validation with {} folds:'.format(cv_fold))

    from numpy import array, amin, mean 
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

        # calculate the cluster centroid
        # NOTE: different definitions of centroids may be used
        cluster_centroids = list()
        for c in clusters:
            cluster_centroids.append(mean([X.loc[ix] for ix in c], axis=0))
        cluster_centroids = array(cluster_centroids) 

        # evaluate using the test set
        sum_closest_proximity = 0.0
        for ix in test_set_index:
            x = X.loc[ix].values.reshape((1, len(X.loc[ix])))
            sum_closest_proximity += amin(
                    cdist(x, cluster_centroids, metric=proximity_metric))
        scores.append((-1) * sum_closest_proximity)

    return mean(scores)


def grid_search(func_core, params_config, func_eval_score):
    """This method provides a wrapper with grid search functionality. 
    
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
    from itertools import product
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
    
    from functools import partial
    from multiprocessing import Pool
    from operator import itemgetter

    with Pool() as p:
        params_best = max(
            p.map(partial(_grid_search_wrapper, 
                    func_core,
                    func_eval_score), 
                l_dicts_all_configs),
            key=itemgetter(1))[0]

    solution = func_core(**params_best)
    return solution, params_best


def _grid_search_wrapper(func_core, func_eval_score, params):
    func_core_ret = func_core(**params)
    score = func_eval_score(func_core_ret)
    return params, score


@deprecated(reason='This method is neither being nor intended to be used.')
def _powerset_exclude_headtail(s, reverse=False, depth=None):
    """Python recipe: this function returns a power set of a given set
    of elements, but excluding the empty set and the given set itself,
    as a generator.

    Parameters
    ----------
    s : set or frozenset
        A given set of elements. 
    reverse : bool, optional, default True
        A boolean flag determining whether the generated power set (as a 
        generator) delivers sets with lower cardinality first or higher 
        ones. Defaults to ``True``, i.e. the lower ones before the 
        higher.
    depth : int, optional, default None
        The upper bound (or lower bound) of cardinality that filters the 
        sets to be generated. Defaults to ``None``, i.e. the whole power
        set will be returned.

    Returns
    -------
    generator
        A power set generated.
    """
    from itertools import chain, combinations
    s = list(s)
    if reverse:
        end = 0 if depth is None else (len(s) - 1 - depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(len(s) - 1, end, -1)))
    else:
        end = len(s) if depth is None else (1 + depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(1, end)))


@deprecated(reason='This method is neither being nor intended to be used.')
def _find_best_subset_GA(universe, evaluate, seed,
    max_iter, size_population, p_crossover, p_mutate):
    from random import randint, sample, random
    from deap import base, creator, tools

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    '''
    # Non-guided initialization
    toolbox.register('attr_bool', randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
            toolbox.attr_bool, len(universe))
    '''

    # use the seed to guide the initialization of individuals
    def initIndividualBySeed(icls, seed):
        init = [0] * len(universe)
        num_activated = randint(1, len(seed))
        index_activated = map(universe.index, sample(seed, num_activated))
        for i in index_activated:
            init[i] = 1
        return icls(init)
    toolbox.register('individual', initIndividualBySeed,
            creator.Individual, seed)

    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', evaluate)
    toolbox.register('crossover', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    while True:
        pop = toolbox.population(n=size_population)
        pop = list(filter(lambda x: any(x) and not all(x), pop))
        for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
            ind.fitness.values = fit
        fits = [ind.fitness.values[0] for ind in pop]

        generation = 0
        while generation < max_iter:
            #print('-' * 5 + 'Generation {}'.format(generation))
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for childx, childy in zip(offspring[::2], offspring[1::2]):
                if random() < p_crossover:
                    toolbox.crossover(childx, childy)
                    del childx.fitness.values
                    del childy.fitness.values

            for child in offspring:
                if random() < p_mutate:
                    toolbox.mutate(child)
                    del child.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind = list(
                    filter(lambda x: any(x) and not all(x), invalid_ind))
            for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
                ind.fitness.values = fit
            
            offspring = list(
                    filter(lambda x: any(x) and not all(x), offspring))
            if len(offspring) == 0:
                # if no valid offspring is to be generated
                break
            else:
                pop[:] = offspring
            generation += 1
        
        top_results = tools.selBest(pop, 1)
        if len(top_results) > 0:
            return frozenset(universe[i] 
                    for i, flag in enumerate(top_results[0]) if flag == 1)
        else:
            pass

