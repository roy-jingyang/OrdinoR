"""
Importing event logs from XES (eXtensible Event Stream) files

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
pm4py.read_xes :
    Parse an XES event log file as an EventLog in pm4py. 
"""

import pm4py

from ordinor.utils.validation import check_required_attributes
from ordinor.utils.converter import convert_to_dataframe

from ._helpers import _describe_event_log

def read_xes(filepath):
    """
    Import an event log from a file in IEEE XES (eXtensible Event 
    Stream) format.

    Parameters
    ----------
    filepath : str or path object
        File path to the event log to be imported.

    Returns
    -------
    el : EventLog
        An event log.
    """
    print(f'Importing from XES file {filepath}')
    event_log = pm4py.read_xes(filepath)

    el = convert_to_dataframe(event_log)
    
    check_required_attributes(el)
    _describe_event_log(el)
    return el