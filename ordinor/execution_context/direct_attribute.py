"""
Several basic execution context learning approaches that directly exploit
event log attributes, namely

    - ATonlyMiner (AT only)
    - CTonlyMiner (CT only)
    - ATCTMiner (CT & AT)
    - ATTTMiner (AT & TT)
    - FullMiner (CT & AT & TT)

"""

from operator import attrgetter

import ordinor.exceptions as exc
from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const

from .base import BaseMiner

class ATonlyMiner(BaseMiner):
    """
    Each value of activity (task) label is taken as an activity type.
    """

    def __init__(self, el):
        BaseMiner.__init__(self, el)


    def _build_atypes(self, el):
        el = check_convert_input_log(el)
        self._atypes = dict()
        for activity_label in set(el[const.ACTIVITY]):
            self._atypes[activity_label] = 'AT.{}'.format(activity_label)

        self.is_atypes_verified = self._verify_partition(
            set(el[const.ACTIVITY]), self._atypes)


class CTonlyMiner(BaseMiner):
    """
    Each value of a selected case-level attribute is taken as a case
    type.
    """
    
    def __init__(self, el, case_attr_name):
        self._build_ctypes(el, case_attr_name)
        BaseMiner._build_atypes(self, el)
        BaseMiner._build_ttypes(self, el)
        self._verify()


    def _build_ctypes(self, el, case_attr_name):
        el = check_convert_input_log(el)
        self._ctypes = dict()
        for v, events in el.groupby(case_attr_name): # sorted by default
            for case_id in set(events[const.CASE_ID]):
                if case_id in self._ctypes:
                    raise exc.InvalidParameterError(
                        param='case_attr_name',
                        reason=f'Not a case-level attribute (check case {case_id})'
                    )
                else:
                    self._ctypes[case_id] = 'CT.{}'.format(v)

        self.is_ctypes_verified = self._verify_partition(
            set(el[const.CASE_ID]), self._ctypes)


class ATCTMiner(ATonlyMiner, CTonlyMiner):
    """
    Each value of activity (task) label is taken as an activity type, 
    and each value of a selected case-level attribute is taken as a case 
    type.
    """

    def __init__(self, el, case_attr_name):
        CTonlyMiner._build_ctypes(self, el, case_attr_name)
        ATonlyMiner._build_atypes(self, el)
        ATonlyMiner._build_ttypes(self, el)
        self._verify()


class ATTTMiner(ATonlyMiner):
    """
    Each value of activity (task) label is taken as an activity type, 
    and each possible value of a designated datetime unit is taken as a 
    time type.
    """

    def __init__(self, el, resolution):
        ATonlyMiner._build_ctypes(self, el)
        ATonlyMiner._build_atypes(self, el)
        self._build_ttypes(el, resolution)
        self._verify()


    def _build_ttypes(self, el, resolution):
        el = check_convert_input_log(el)
        self._ttypes = dict()
        if resolution in ['weekday', 'isoweekday']:
            # special case since (iso)weekday is a function of datetime
            # parse as 'isoweekday', i.e. Monday - 1, ..., Sunday - 7
            getter = (
                lambda datetimeobj: const.WEEKDAYS[datetimeobj.weekday()]
            )
        else:
            # otherwise access as read-only attribute
            getter = attrgetter(resolution)

        for event in el.to_dict(orient='records'):
            dt = event[const.TIMESTAMP]
            self._ttypes[event[const.TIMESTAMP]] = 'TT.{}'.format(getter(dt))

        self.is_ttypes_verified = self._verify_partition(
            set(el[const.TIMESTAMP]), self._ttypes)


class FullMiner(CTonlyMiner, ATTTMiner):
    """
    Each value of activity (task) label is taken as an activity type, 
    each value of a selected case-level attribute is taken as a case 
    type, and each possible value of a designated datetime unit is taken 
    as a time type.
    """

    def __init__(self, el, 
        case_attr_name, resolution):
        CTonlyMiner._build_ctypes(self, el, case_attr_name)
        ATTTMiner._build_atypes(self, el)
        ATTTMiner._build_ttypes(self, el, resolution)
        self._verify()
