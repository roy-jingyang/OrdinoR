"""
Converting between different data types
"""

import pm4py.convert

def convert_to_dataframe(event_log):
    """
    This is an alias of the pm4py method ``convert_to_dataframe``.
    
    Convert an EventLog instance into a pm4py-compatible pandas DataFrame.

    Parameters
    ----------
    event_log : EventLog
        An event log.

    Returns
    -------
    pandas.DataFrame
        An event log.

    See Also
    --------
    pm4py.convert.convert_to_dataframe
    """
    return pm4py.convert.convert_to_dataframe(event_log)
