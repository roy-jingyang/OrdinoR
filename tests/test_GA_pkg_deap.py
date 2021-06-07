#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import random

def _is_odd_number(x):
    return 1 if x % 2 == 1 else float('-inf')

from deap import base, creator, tools
def find_best_subsets(universe):
    print(universe)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register('attr_bool', random.randint, 0, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual,
            toolbox.attr_bool, len(universe))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        return (sum(_is_odd_number(universe[i]) 
            for i, flag in enumerate(individual) if flag == 1),)
    toolbox.register('evaluate', evaluate)
    toolbox.register('crossover', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=3)

    pop = toolbox.population(n=300)
    CXPB, MUTPB = 0.5, 0.2

    print('Start evolution')
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit
    fits = [ind.fitness.values[0] for ind in pop]

    g = 0
    while g < 1000:
        g += 1
        #print('-' * 5 + 'Generation {}'.format(g) + '-' * 5)

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for childx, childy in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.crossover(childx, childy)
                del childx.fitness.values
                del childy.fitness.values

        for child in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(child)
                del child.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)):
            ind.fitness.values = fit

        pop[:] = offspring

    best_ind = tools.selBest(pop, 1)[0]
    return list(universe[i] for i, flag in enumerate(best_ind) if flag == 1)


if __name__ == '__main__':
    u = list(range(10))
    sol = find_best_subsets(u)
    print(sol)
