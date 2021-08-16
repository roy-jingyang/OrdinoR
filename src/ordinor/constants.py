"""
Constants shared across all modules in OrdinoR
"""

# standard event attribute names in OrdinoR, following pm4py
from pm4py.util import constants
from pm4py.util import xes_constants

CASE_ID         = constants.CASE_CONCEPT_NAME
ACTIVITY        = xes_constants.DEFAULT_NAME_KEY
TIMESTAMP       = xes_constants.DEFAULT_TIMESTAMP_KEY
TIMESTAMP_ST    = xes_constants.DEFAULT_START_TIMESTAMP_KEY
RESOURCE        = xes_constants.DEFAULT_RESOURCE_KEY
GROUP           = xes_constants.DEFAULT_GROUP_KEY

# additional event attribute names

CASE_DURATION   = 'case_duration_seconds'

# resource log attribute names

CASE_TYPE       = 'case_type'
ACTIVITY_TYPE   = 'activity_type'
TIME_TYPE       = 'time_type'

# ISO weekday names
WEEKDAYS        = (
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
)


# Fluxicon Disco interface
class DISCO_KEYS:
    """
    This class contains Fluxicon Disco exported event attribute names
    (used in CSV files).
    """
    CASE_ID         = 'Case ID'
    ACTIVITY        = 'Activity'
    TIMESTAMP       = 'Complete Timestamp'
    TIMESTAMP_ST    = 'Start Timestamp'
    RESOURCE        = 'Resource'

    TIMESTAMP_FMT_STR   = '%Y-%m-%d %H:%M:%S.%f'

