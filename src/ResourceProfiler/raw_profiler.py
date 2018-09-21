# -*- coding: utf-8 -*-

'''
This module contains the implementation of profiling a resource using the 
"raw" information in the given event log. Methods include:
'''

# Note: this is a redundant copy of 
#   SocialNetworkMiner.mining.joint_activities.build_performer_activity_matrix
# TODO: when import issue is solved, the redundancy should be removed.
def performer_activity_frequency(c, use_log_scale):
    '''
    This method builds a "profile" based on how frequent individuals conduct
    specific activities. The "performer by activity matrix" is used to repre-
    sent these profiles.

    Params:
        c: DataFrame
            The imported event log.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        pam: DataFrame
            The constructed performer by activity matrix as a pandas DataFrame,
            with resource ids as indices and activity names as columns.
    '''

    from collections import defaultdict
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for case_id, trace in c.groupby('case_id'):
        for event in trace.itertuples():
            pam[event.resource][event.activity] += 1

    from pandas import DataFrame
    if use_log_scale: 
        from numpy import log
        return DataFrame.from_dict(pam, orient='index').fillna(0).apply(
                lambda x: log(x + 1))
    else:
        return DataFrame.from_dict(pam, orient='index').fillna(0)

