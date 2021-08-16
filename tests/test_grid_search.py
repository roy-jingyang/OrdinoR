#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# import methods to be tested below
from functools import partial

from ordinor.org_model_miner._helpers import grid_search

# List input parameters from shell

if __name__ == '__main__':
    def foo(x, y):
        return (x + y) ** 2 + 2 * (x + y) + 1
        #return (x + y) * 10

    def foo_score(foo_ret):
        return foo_ret / 10
    
    solution, params_best = grid_search(
        partial(foo, x=0),
        params_config={
            #'x': list(range(1, 11)),
            'y': list(range(-3, 2)) 
        },
        func_eval_score=foo_score
    )

    print(solution)
    print(params_best)