# -*- coding: utf-8 -*-

def append_case_duration(el, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
    from datetime import datetime
    l_case_duration = [float('nan')] * len(el)
    for case_id, events in el.groupby('case_id'):
        start_time = datetime.strptime(
            events.iloc[0]['timestamp'], datetime_format)
        complete_time = datetime.strptime(
            events.iloc[-1]['timestamp'], datetime_format)
        duration_seconds = (complete_time - start_time).total_seconds()
        for event_index in events.index:
            l_case_duration[event_index] = duration_seconds

    el['case_duration'] = l_case_duration
    return el

