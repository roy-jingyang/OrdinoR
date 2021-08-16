"""
Construct an individual resource feature matrix given an event log (or
the corresponding resource log)
"""

from collections import defaultdict

from pandas import DataFrame
from numpy import log

import ordinor.exceptions as exc
import ordinor.constants as const

def direct_count(rl, scale=None):
    """
    Build resource feature matrix based on how frequently resources 
    originated events of different execution contexts.
    
    Each column in the result profiles corresponds with an execution
    context captured in the given resource log.

    Parameters
    ----------
    rl : pandas.DataFrame
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
        The constructed resource profile matrix.
    """
    mat = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in rl.groupby(const.RESOURCE):
        for event in trace.to_dict(orient='records'):
            exe_ctx = (
                event[const.CASE_TYPE], 
                event[const.ACTIVITY_TYPE], 
                event[const.TIME_TYPE]
            )
            mat[res][exe_ctx] += 1

    df = DataFrame.from_dict(mat, orient='index').fillna(0)
    if scale is None:
        return df
    elif scale == 'normalize':
        return df.div(df.sum(axis=1), axis=0)
    elif scale == 'log':
        # NOTE: log_e(x + 1)
        return df.apply(lambda x: log(x + 1))
    else:
        raise exc.InvalidParameterError(
            param='scale',
            reason='Can only be one of {"normalize", "log", None}'
        )
