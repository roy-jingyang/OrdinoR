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
    scale : {None, 'workload', 'logarithm', 'standardize'}, optional,
        default None
        Options for deciding how to scale the values of frequency
        counting. Could be one of the following:
            
            - ``None``, no scaling will be performed.
            - ``'workload'``, scale the frequency values by the total
              count of executions by each resource (scale by row).
            - ``'logarithm'``, scale the frequency values by logarithm.
            - ``'standardize'``, scale the values by removing the mean and scaling to unit variance.

    Returns
    -------
    DataFrame
        The constructed resource profiles.
    
    Raises
    ------
    ValueError
        If the specified option for scaling is invalid.

    See Also
    --------
    sklearn.preprocessing.scale
    """
    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for res, events in rl.groupby('resource'):
        for event in events.itertuples():
            exec_mode = (event.case_type, event.activity_type, event.time_type)
            mat[res][exec_mode] += 1

    from pandas import DataFrame
    df = DataFrame.from_dict(mat, orient='index').fillna(0)
    if scale is None:
        return df
    elif scale == 'workload':
        # scale by row (resource)
        return df.div(df.sum(axis=1), axis=0)
    elif scale == 'logarithm':
        # f(x) = log_e(x + 1)
        from numpy import log 
        return df.applymap(lambda x: log(x + 1))
    elif scale == 'standardize':
        # standardization: z = (x - u) / s
        from sklearn.preprocessing import scale
        return df.apply(scale, raw=True)
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'scale', scale))


def split_count_execution_frequency(rl, scale=None):
    """Build resource profiles based on how frequently resources 
    originated events of execution modes.
    
    Each column in the result profiles corresponds with an execution
    mode captured in the given resource log.

    Parameters
    ----------
    rl : DataFrame
        A resource log.
    scale : {None, 'workload', 'logarithm', 'standardize'}, optional,
        default None
        Options for deciding how to scale the values of frequency
        counting. Could be one of the following:
            
            - ``None``, no scaling will be performed.
            - ``'standardize'``, scale the values by removing the mean and scaling to unit variance.

    Returns
    -------
    DataFrame
        The constructed resource profiles.
    
    Raises
    ------
    ValueError
        If the specified option for scaling is invalid.

    See Also
    --------
    sklearn.preprocessing.scale
    """
    from collections import defaultdict
    mat = defaultdict(lambda: defaultdict(lambda: 0))

    for res, events in rl.groupby('resource'):
        for event in events.itertuples():
            if event.activity_type != '':
                mat[res][event.activity_type] += 1
            if event.time_type != '':
                mat[res][event.time_type] += 1
    # TODO: log group by cases (for counting CT features)

    from pandas import DataFrame
    df = DataFrame.from_dict(mat, orient='index').fillna(0)
    if scale is None:
        return df
    elif scale == 'standardize':
        # standardization: z = (x - u) / s
        from sklearn.preprocessing import scale
        return df.apply(scale, raw=True)
    else:
        raise ValueError('Invalid value for parameter `{}`: {}'.format(
            'scale', scale))
