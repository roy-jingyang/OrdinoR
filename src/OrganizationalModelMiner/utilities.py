# -*- coding: utf-8 -*-

'''
This module provides the necessary helper functions for organizational model
mining methods.
'''

def cross_validation_score(
        X, miner, miner_params,
        proximity_metric='euclidean', cv_fold=0.2):
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
    # TODO
    #from numpy.random import shuffle
    #shuffle(index)
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

def powerset_exclude_headtail(iterable, reverse=False, depth=None):
    '''
    Python recipe: this is a helper function
    This function returns a generator Powerset(s) \ {emptyset, s} given a set s

    Params:
        iterable: iterable
        reverse: Boolean, optional
            The generator delivers subsets based on cardinality on an ascending
            order, specify the additional argument 'reverse' to change the
            behaviour.
        depth: int, optional
            The additional argument 'depth' specifies the maximal (or minimal) 
            cardinality of subset(s) returned by the function. If None, all
            will be returned.
    Returns:
        generator
            The subsets in the powerset as required.
    '''
    from itertools import chain, combinations
    s = list(iterable)
    if reverse:
        end = 0 if depth is None else (len(s) - 1 - depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(len(s) - 1, end, -1)))
    else:
        end = len(s) if depth is None else (1 + depth)
        return (chain.from_iterable(combinations(s, r) 
            for r in range(1, end)))

def find_best_subset_GA(universe, evaluate,
        max_iter, size_population, p_crossover, p_mutate):
        from random import randint, random
        from deap import base, creator, tools

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register('attr_bool', randint, 0, 1)
        toolbox.register('individual', tools.initRepeat, creator.Individual,
                toolbox.attr_bool, len(universe))
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

