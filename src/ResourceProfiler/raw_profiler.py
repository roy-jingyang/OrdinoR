# -*- coding: utf-8 -*-

'''
This module contains the implementation of profiling a resource using the 
"raw" information in the given resource log. Methods include:
'''

def performer_activity_frequency(rl, use_log_scale):
    '''
    This method builds a "profile" based on how frequent individuals originate
    events with specific activity names, i.e. the performer-by-activity matrix.

    Params:
        rl: DataFrame
            The resource log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        pam: DataFrame
            The constructed performer by activity matrix as a pandas DataFrame,
            with resource ids as indices and activity names as columns.
        X: DataFrame
            The contructed resource profiles.
    '''

    from collections import defaultdict
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in rl.groupby('resource'):
        for event in trace.itertuples():
            pam[res][event.activity_type] += 1

    from pandas import DataFrame
    if use_log_scale: 
        from numpy import log
        return DataFrame.from_dict(pam, orient='index').fillna(0).apply(
                lambda x: log(x + 1))
    else:
        return DataFrame.from_dict(pam, orient='index').fillna(0)

