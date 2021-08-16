"""
Validating inputs
"""

from pm4py.objects.log.obj import EventLog
from pandas import DataFrame

import ordinor.constants as const
import ordinor.exceptions as exc

from .converter import convert_to_dataframe

def check_required_attributes(el):
    """
    Check if an event log has the required event attributes.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log.

    Returns
    -------
    """
    has_required_attributes = (
        const.CASE_ID in el.columns and
        const.ACTIVITY in el.columns and
        const.TIMESTAMP in el.columns and
        const.RESOURCE in el.columns
    )
    if has_required_attributes:
        return
    else:
        raise exc.DataMissingError(
            f"""
            One or more of the required event attributes (case id,
            activity label, timestamp, resource) are not found:
            expected dataframe columns ["{const.CASE_ID}",
            "{const.ACTIVITY}", "{const.TIMESTAMP}", "{const.RESOURCE}"].

            Are there missing or duplicate data attributes in the
            original data?
            """
        )


def check_convert_input_log(el):
    """
    Check if an input event log is a pandas DataFrame with the required
    event attributes.

    If it is a pm4py EventLog, convert it to a compatible dataframe and
    check the attributes. 

    Parameters
    ----------
    el : EventLog
        An event log.

    Returns
    -------
    pandas.DataFrame or None
        An event log (as a pandas DataFrame) with the required event
        attributes; or None if the validation fails with errors raised.
    """
    if type(el) is DataFrame:
        pass
    elif type(el) is EventLog:
        el = convert_to_dataframe(el)
    else:
        raise exc.InvalidParameterError(
            param='el',
            reason='Expected a pandas DataFrame or a pm4py EventLog'
        )
    
    check_required_attributes(el)
    return el
