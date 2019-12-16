# -*- coding: utf-8 -*-

"""This module contains methods for augmenting (extending) an event log
(as DataFrame) using the original information recorded in the log. 

For example, obtain the duration time of cases by calculating the time 
difference between the first and the last events in each case.
"""
def append_case_duration(el, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
    """Calculate and append case duration information to cases in a 
    given event log.

    Parameters
    ----------
    el : DataFrame
        An event log.
    datetime_format : str, optional, default '%Y/%m/%d %H:%M:%S.%f'
        The format string for parsing the timestamps in the given event
        log. Defaults to ``'%Y/%m/%d %H:%M:%S.%f'``, e.g. 2019/12/6 
        10:48:51.0. See `Python strftime() and strptime() Behavior` for 
        more information.

    Returns
    el : DataFrame
        An event log.
    """
    from datetime import datetime
    l_case_duration = [float('nan')] * len(el)
    for case_id, trace in el.groupby('case_id'):
        start_time = datetime.strptime(
            trace.iloc[0]['timestamp'], datetime_format)
        complete_time = datetime.strptime(
            trace.iloc[-1]['timestamp'], datetime_format)
        duration_seconds = (complete_time - start_time).total_seconds()
        for event_index in trace.index:
            l_case_duration[event_index] = duration_seconds

    el['case_duration'] = l_case_duration
    return el

