"""
Preprocessing an event log (as a DataFrame)
"""

from datetime import datetime

import pm4py

from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const
import ordinor.exceptions as exc

def format_dataframe(df,
                     case_id,
                     activity_key,
                     timestamp_key,
                     start_timestamp_key,
                     resource_id,
                     timest_format=None):
    """
    Format a pandas DataFrame by renaming the columns to compatible
    names.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas dataframe to be formatted.
    case_id : str
        Column name for case ids.
    activity_key : str
        Column name for activity labels.
    timestamp_key : str
        Column name for timestamps.
    start_timestamp_key : str
        Column name for start timestamps.
    resource_id : str
        Column name for resource ids.
    timest_format : str, optional
        Timestamp format. Inferred if not provided.
    """
    df = pm4py.format_dataframe(
        df,
        case_id=case_id,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        start_timestamp_key=start_timestamp_key,
        timest_format=timest_format
    )

    if resource_id not in df.columns:
        raise exc.DataMissingError(
            f'{resource_id} is not in the columns.'
        )
    else:
        df = df.rename(columns={resource_id: const.RESOURCE})
    
    return df


def append_case_duration(el):
    """
    Calculate and append case duration time (in seconds) to cases in a
    given event log.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log.

    Returns
    el : pandas.DataFrame
        An event log.
    """
    el = check_convert_input_log(el)
    print(el)
    l_case_duration = [float('nan')] * len(el)
    for case_id, trace in el.groupby(const.CASE_ID):
        start_time = trace.iloc[0][const.TIMESTAMP]
        complete_time = trace.iloc[-1][const.TIMESTAMP]
        duration_seconds = (complete_time - start_time).total_seconds()
        for event_index in trace.index:
            l_case_duration[event_index] = duration_seconds

    el[const.CASE_DURATION] = l_case_duration
    return el

