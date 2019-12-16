# -*- coding: utf-8 -*-

"""This module provides several simple execution mode learning 
approaches, namely

    - ATonlyMiner (AT only)
    - CTonlyMiner (CT only)
    - ATCTMiner (CT & AT)
    - ATTTMiner (AT & TT)
    - FullMiner (CT & AT & TT)

"""
from .base import BaseMiner

class ATonlyMiner(BaseMiner):
    """Each value of activity (task) label is taken as an activity type.
    """

    def __init__(self, el):
        BaseMiner.__init__(self, el)


    def _build_atypes(self, el):
        self._atypes = dict()
        for activity_label in set(el['activity']):
            self._atypes[activity_label] = 'AT.{}'.format(activity_label)

        self.is_atypes_verified = self._verify_partition(
            set(el['activity']), self._atypes)


class CTonlyMiner(BaseMiner):
    """Each value of a selected case-level attribute is taken as a case
    type.

    Raises
    ------
    ValueError
        If the specified case-level attribute does not qualify, i.e., 
        events from the same case record more than one values for this 
        attribute. Case id of the first problematic case will be 
        reported.

    """
    
    def __init__(self, el, case_attr_name):
        self._build_ctypes(el, case_attr_name)
        BaseMiner._build_atypes(self, el)
        BaseMiner._build_ttypes(self, el)
        self._verify()


    def _build_ctypes(self, el, case_attr_name):
        self._ctypes = dict()
        for v, events in el.groupby(case_attr_name): # sorted by default
            for case_id in set(events['case_id']):
                if case_id in self._ctypes:
                    raise ValueError('Inappropriate selection for ' +
                        '`case_attr_name`: check case id {}.'.format(case_id))
                else:
                    self._ctypes[case_id] = 'CT.{}'.format(v)

        self.is_ctypes_verified = self._verify_partition(
            set(el['case_id']), self._ctypes)


class ATCTMiner(ATonlyMiner, CTonlyMiner):
    """Each value of activity (task) label is taken as an activity type, 
    and each value of a selected case-level attribute is taken as a case 
    type.
    """

    def __init__(self, el, case_attr_name):
        CTonlyMiner._build_ctypes(self, el, case_attr_name)
        ATonlyMiner._build_atypes(self, el)
        ATonlyMiner._build_ttypes(self, el)
        self._verify()


class ATTTMiner(ATonlyMiner):
    """Each value of activity (task) label is taken as an activity type, 
    and each possible value of a designated datetime unit is taken as a 
    time type.
    """

    def __init__(self, el, resolution, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
        ATonlyMiner._build_ctypes(self, el)
        ATonlyMiner._build_atypes(self, el)
        self._build_ttypes(el, resolution, datetime_format)
        self._verify()


    def _build_ttypes(self, el, resolution, datetime_format):
        self._ttypes = dict()
        from datetime import datetime
        if resolution in ['weekday', 'isoweekday']:
            # special case since (iso)weekday is a function of datetime
            # parse as 'isoweekday', i.e. Monday - 1, ..., Sunday - 7
            getter = lambda datetimeobj: datetimeobj.isoweekday()
        else:
            # otherwise access as read-only attribute
            from operator import attrgetter
            getter = attrgetter(resolution)

        for event in el.itertuples():
            dt = datetime.strptime(event.timestamp, datetime_format)
            self._ttypes[event.timestamp] = 'TT.{}'.format(getter(dt))

        self.is_ttypes_verified = self._verify_partition(
            set(el['timestamp']), self._ttypes)


class FullMiner(CTonlyMiner, ATTTMiner):
    """Each value of activity (task) label is taken as an activity type, 
    each value of a selected case-level attribute is taken as a case 
    type, and each possible value of a designated datetime unit is taken 
    as a time type.
    """

    def __init__(self, el, 
        case_attr_name, resolution, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
        CTonlyMiner._build_ctypes(self, el, case_attr_name)
        ATTTMiner._build_atypes(self, el)
        ATTTMiner._build_ttypes(self, el, resolution, datetime_format)
        self._verify()

