"""
Importing event logs from CSV (Comma-Separated Values) files

Notes
-----
The input event log data should at least provide the information of 
case ids, activity labels, timestamps, and resource ids.

By default, all input event logs will be imported as pandas DataFrames in
OrdinoR. To import event logs as pm4py EventLog instances, please use the
original pm4py utilities.

See Also
--------
pandas.DataFrame : 
    The primary pandas data structure.
pm4py.objects.log.obj.EventLog : 
    An event log in pm4py.
pm4py.format_dataframe : 
    Return a formatted copy of a pandas DataFrame, compatible with pm4py.
"""

import pandas as pd

import ordinor.constants as const
import ordinor.exceptions as exc
from ordinor.utils.log_preprocessing import format_dataframe
from ordinor.utils.validation import check_required_attributes

from ._helpers import _describe_event_log

def _drop_duplicate_csv_columns(df):
    num_columns = len(df.columns)
    filtered_df = df.drop(
        columns=[const.CASE_ID,
                 const.ACTIVITY,
                 const.TIMESTAMP,
                 const.TIMESTAMP_ST
        ],
        errors='ignore'
    )
    if len(filtered_df.columns) < num_columns:
        exc.warn_import_data_ignored(
            'Column names conflict with reserved event attribute names.'
        )
    return filtered_df


def read_disco_csv(filepath):
    """
    Import an event log from a file in CSV (Column-Separated Values)
    format exported from Fluxicon Disco.

    Parameters
    ----------
    filepath : str or path object
        File path to the event log to be imported.

    See Also
    --------
    ordinor.io.reader.read_csv

    Returns
    -------
    el : pandas.DataFrame
        An event log.
    """

    return read_csv(filepath,
                    sep=',',
                    case_id=const.DISCO_KEYS.CASE_ID,
                    activity_key=const.DISCO_KEYS.ACTIVITY,
                    timestamp_key=const.DISCO_KEYS.TIMESTAMP,
                    start_timestamp_key=const.DISCO_KEYS.TIMESTAMP_ST,
                    timest_format=const.DISCO_KEYS.TIMESTAMP_FMT_STR,
                    resource_id=const.DISCO_KEYS.RESOURCE)

def read_csv(filepath, 
             sep=',', 
             case_id=const.CASE_ID,
             activity_key=const.ACTIVITY,
             timestamp_key=const.TIMESTAMP,
             start_timestamp_key=const.TIMESTAMP_ST,
             resource_id=const.RESOURCE,
             timest_format=None,
             **pdkwargs):
    """
    Import an event log from a file in CSV (Column-Separated Values)
    format.

    The CSV file is expected to contain at least four default columns:

        - case id,
        - activity label,
        - complete timestamp,
        - resource id.
    
    The expected column names follow pm4py definitions.

    Parameters
    ----------
    filepath : str or path object
        File path to the event log to be imported.
    sep : str
        Column separator. Default is ``,``.
    case_id : str, optional
        Column name for case ids.
    activity_key : str, optional
        Column name for activity labels.
    timestamp_key : str, optional
        Column name for timestamps.
    start_timestamp_key : str, optional
        Column name for start timestamps.
    timest_format : str, optional
        Timestamp format. Inferred if not provided.
    **pdkwargs : additional keyword parameters, optional
        Additional parameters to be passed to ``pandas.read_csv``.

    See Also
    --------
    pandas.read_csv : 
        Read a comma-separated values (csv) file into a DataFrame.

    Returns
    -------
    el : pandas.DataFrame
        An event log.
    """
    print(f'Importing from CSV file {filepath}')
    df = pd.read_csv(filepath, sep=sep, **pdkwargs)
    df = _drop_duplicate_csv_columns(df)

    print(f'Scanned {len(df)} events from "{filepath}".')

    el = format_dataframe(
        df,
        case_id=case_id,
        activity_key=activity_key,
        timestamp_key=timestamp_key,
        start_timestamp_key=start_timestamp_key,
        resource_id=resource_id,
        timest_format=timest_format,
    )

    check_required_attributes(el)
    _describe_event_log(el)
    return el
