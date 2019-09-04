# -*- coding: utf-8 -*-

'''
This module contains the implementations of some naive execution mode learning
methods that derive C/A/T types directly based on grouping over attribute 
values accessible in the log data:
    - ATonlyMiner (AT only)
    - ATCTMiner (AT + CT)
    - ATTTMiner (AT + TT)
'''

from .base import BaseMiner

class ATonlyMiner(BaseMiner):
    '''(AT only method) Each value of activity (task) label is taken as
    an activity type, ignoring the case and time dimensions.
    '''

    def _build_ctypes(self, el):
        # ignoring the case dimension: all events are of the same case type.
        self.is_ctypes_verified = True
        pass
    
    def _build_atypes(self, el):
        self._atypes = dict()
        for event in el.itertuples():
            self._atypes[event.activity] = 'AT.{}'.format(event.activity)

        self.is_atypes_verified = self.verify_partition(
            set(el['activity']), self._atypes)

    def _build_ttypes(self, el):
        # ignoring the time dimension: all events are of the same time type.
        self.is_ttypes_verified = True
        pass

    def derive_resource_log(self, el):
        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            if event.resource != '' and event.resource is not None:
                rl.append({
                    'resource': event.resource,
                    'case_type': '',
                    'activity_type': self._atypes[event.activity],
                    'time_type': ''
                })

        from pandas import DataFrame
        return DataFrame(rl)

class CTonlyMiner(BaseMiner):
    '''(CT only method) The attribute values of events and their cases are used
    to derive case types, ignoring the activity and time dimensions.
    '''
    
    def __init__(self, el, case_attr_name):
        self._build_ctypes(el, case_attr_name)
        self._build_atypes(el)
        self._build_ttypes(el)
        self.verify()

    def _build_ctypes(self, el, case_attr_name):
        self._ctypes = dict()
        par = list()
        # 1) directly let each value correspond to a category (type)
        for v, events in el.groupby(case_attr_name): # sorted by default
            par.append(set(events['case_id']))

        '''
        # 2) another way: use bins to convert continuous values to categorical
        # NOTE: BPIC12 case (AMOUNT_REQ): from 0 to 50k, seg size 5k; + others
        # 11 bins in total
        from pandas import cut
        for v, events in el.groupby(cut(el[case_attr_name].astype('int'), 
            bins=list(range(-1, 50000, 5000)) + [100000])):
            par.append(set(events['case_id']))
        '''
        for i, values in enumerate(par):
            for v in values:
                self._ctypes[v] = 'CT.{}'.format(i)

        self.is_ctypes_verified = self.verify_partition(
            set(el['case_id']), self._ctypes)

    def _build_atypes(self, el):
        # ignoring the activity dimension: all events are of the same act type.
        self.is_atypes_verified = True
        pass

    def _build_ttypes(self, el):
        # ignoring the time dimension: all events are of the same time type.
        self.is_ttypes_verified = True
        pass

    def derive_resource_log(self, el):
        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            if event.resource != '' and event.resource is not None:
                rl.append({
                    'resource': event.resource,
                    'case_type': self._ctypes[event.case_id],
                    'activity_type': '',
                    'time_type': ''
                })

        from pandas import DataFrame
        return DataFrame(rl)

class ATCTMiner(ATonlyMiner):
    '''(AT + CT method) Based on ATonlyMiner (AT only), let each value of a
    designated case attribute in the events be a case type, ignoring the time
    dimension.
    '''
    
    def __init__(self, el, case_attr_name):
        self._build_ctypes(el, case_attr_name)
        self._build_atypes(el)
        self._build_ttypes(el)
        self.verify()

    def _build_ctypes(self, el, case_attr_name):
        self._ctypes = dict()
        par = list()
        # 1) directly let each value correspond to a category (type)
        for v, events in el.groupby(case_attr_name): # sorted by default
            par.append(set(events['case_id']))

        '''
        # 2) another way: use bins to convert continuous values to categorical
        # NOTE: BPIC12 case (AMOUNT_REQ): from 0 to 50k, seg size 5k; + others
        # 11 bins in total
        from pandas import cut
        for v, events in el.groupby(cut(el[case_attr_name].astype('int'), 
            bins=list(range(-1, 50000, 5000)) + [100000])):
            par.append(set(events['case_id']))
        '''
        for i, values in enumerate(par):
            for v in values:
                self._ctypes[v] = 'CT.{}'.format(i)

        self.is_ctypes_verified = self.verify_partition(
            set(el['case_id']), self._ctypes)

    def derive_resource_log(self, el):
        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            if event.resource != '' and event.resource is not None:
                rl.append({
                    'resource': event.resource,
                    'case_type': self._ctypes[event.case_id],
                    'activity_type': self._atypes[event.activity],
                    'time_type': ''
                })

        from pandas import DataFrame
        return DataFrame(rl)

class ATTTMiner(ATonlyMiner):
    '''(AT + TT method) Based on ATonlyMiner (AT only), let each possible value
    of a designated datetime unit be a time type, ignoring the case dimension.
    '''
    
    def __init__(self, el, resolution, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
        self._build_ctypes(el)
        self._build_atypes(el)
        self._build_ttypes(el, resolution, datetime_format)
        self.verify()

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

        self.is_ttypes_verified = self.verify_partition(
            set(el['timestamp']), self._ttypes)

    def derive_resource_log(self, el):
        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            if event.resource != '' and event.resource is not None:
                rl.append({
                    'resource': event.resource,
                    'case_type': '',
                    'activity_type': self._atypes[event.activity],
                    'time_type': self._ttypes[event.timestamp],
                })

        from pandas import DataFrame
        return DataFrame(rl)

