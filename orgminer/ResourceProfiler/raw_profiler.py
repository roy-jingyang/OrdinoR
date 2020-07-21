# -*- coding: utf-8 -*-

"""This module contains the implementation of methods for profiling 
resources using information directly accessible in a resource log.
"""
def count_execution_frequency(rl, scale=None):
    """Build resource profiles based on how frequently resources 
    originated events of execution modes.
    
    Each column in the result profiles corresponds with an execution
    mode captured in the given resource log.

    Parameters
    ----------
    rl : DataFrame
        A resource log.
    scale : {None, 'normalize', log'}, optional, default None
        Options for deciding how to scale the values of frequency
        counting. Could be one of the following:
            
            - ``None``, no scaling will be performed.
            - ``'normalize'``, scale the frequency values by the total
              count of executions by each resource (scale by row).
            - ``'log'``, scale the frequency values by logarithm.

    Returns
    -------
    DataFrame
        The constructed resource profiles.
    
    Raises
    ------
    ValueError
        If the specified option for scaling is invalid.
    """
    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in rl.groupby('resource'):
        for event in trace.itertuples():
            exec_mode = (event.case_type, event.activity_type, event.time_type)
            mat[res][exec_mode] += 1

    from pandas import DataFrame
    df = DataFrame.from_dict(mat, orient='index').fillna(0)
    if scale is None:
        return df
    elif scale == 'normalize':
        return df.div(df.sum(axis=1), axis=0)
    elif scale == 'log':
        # NOTE: log_e(x + 1)
        from numpy import log 
        return df.apply(lambda x: log(x + 1))
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'scale', scale))

