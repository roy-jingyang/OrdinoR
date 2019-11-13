#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from skfeature.utility.construct_W import construct_W
from skfeature.function.similarity_based import lap_score

# List input parameters from shell


if __name__ == '__main__':
    from sklearn.datasets import load_iris, load_digits, load_wine

    ds = load_iris()
    #ds = load_digits()
    #ds = load_wine()

    X = ds['data']
    y = ds['target']
    print(X.shape)
    print(y.shape)

    score = lap_score.lap_score(X,
        W=construct_W(X, y=y.reshape(X.shape[0], 1),
            metric='euclidean',
            neighbor_mode='supervised',
            weight_mode='heat_kernel',
            k=2,
            t=1))
    print(score)

    ranking_by_score = lap_score.feature_ranking(score) # ascending
    print('Ranking of columns by feature selection:')
    for i, col_idx in enumerate(ranking_by_score):
        print('Rank-{}: Column {}'.format(i+1, col_idx))
    print('\t(Feature scores range from [{}, {}])'.format(
        score[ranking_by_score[0]],
        score[ranking_by_score[-1]]))

