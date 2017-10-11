#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from collections import defaultdict

def mine(cases):
    print('Applying Default Mining:')
    # entity: task_name => [involved originators]
    entities = defaultdict(lambda: set())
    cnt = 0
    for caseid, trace in cases.items():
        cnt += 1
        for i in range(len(trace)):
            resource = trace[i][2]
            activity = trace[i][3]
            entities[activity].add(resource)
    print('# of cases processed: {}'.format(cnt))
    print('{} organizational entities extracted.'.format(len(entities)))
    return copy.deepcopy(entities)

