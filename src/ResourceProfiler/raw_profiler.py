# -*- coding: utf-8 -*-

'''
This module contains the implementation of profiling a resource using the 
"raw" information in the given resource log. Methods include:
'''

# TODO
def count_execution_frequency(rl, scale=None):
    '''
    This method builds a "profile" based on how frequent individuals originated
    events with specific execution modes.

    Params:
        rl: DataFrame
            The resource log.
        scale: string
            Use whether normalization (div. by total) or logarithm scale. The
            default is None.
                - 'normalize'
                - 'log'
    Returns:
        X: DataFrame
            The contructed resource profiles.
    '''

    from collections import defaultdict
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in rl.groupby('resource'):
        for event in trace.itertuples():
            exec_mode = (event.case_type, event.activity_type, event.time_type)
            pam[res][exec_mode] += 1

    from pandas import DataFrame
    df = DataFrame.from_dict(pam, orient='index').fillna(0)
    if scale is None:
        return df
    elif scale == 'log':
        print('Using logarithm scale for frequencies')
        from numpy import log # NOTE: be careful, this is a "ln"
        return df.apply(lambda x: log(x + 1))
    elif scale == 'normalize':
        print('Using normalization for frequencies')
        return df.div(df.sum(axis=1), axis=0)
    else:
        exit('[Error] Unspecified scaling option')

