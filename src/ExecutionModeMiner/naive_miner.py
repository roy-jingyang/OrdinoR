# -*- coding: utf-8 -*-

'''
This module contains the implementations of some naive execution mode learning
methods:
    - ATonlyMiner (AT only)
    - ATCTMiner (AT + CT)
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
        # designate the partitioning (on A)
        # (map each activity names to itself)
        par = [{a} for a in sorted(el['activity'].unique())]

        # validate the partitioning
        is_disjoint = set.intersection(*par) == set()
        is_union = set.union(*par) == set(el['activity'])

        # build types
        if is_disjoint and is_union:
            for i, values in enumerate(par):
                self._atypes['AT.{}'.format(i)] = values.copy()
            self.is_atypes_verified = True
        else:
            self.is_atypes_verified = False

    def _build_ttypes(self, el):
        # ignoring the time dimension: all events are of the same time type.
        self.is_ttypes_verified = True
        pass

    def derive_resource_log(self, el):
        # construct reversed dict: A -> AT
        rev_atypes = dict()
        for type_name, type_value in self._atypes.items():
            for v in type_value:
                rev_atypes[v] = type_name

        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': '',
                'activity_type': rev_atypes[event.activity],
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
        # designate the partitioning (on C)
        # (map the values of a specified attribute to case_ids)
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

        # validate the partitioning
        is_disjoint = set.intersection(*par) == set()
        is_union = set.union(*par) == set(el['case_id'])

        # build types
        if is_disjoint and is_union:
            for i, values in enumerate(par):
                self._ctypes['CT.{}'.format(i)] = values.copy()
            self.is_ctypes_verified = True
        else:
            self.is_ctypes_verified = False

    def _build_atypes(self, el):
        # ignoring the activity dimension: all events are of the same act type.
        self.is_atypes_verified = True
        pass

    def _build_ttypes(self, el):
        # ignoring the time dimension: all events are of the same time type.
        self.is_ttypes_verified = True
        pass

    def derive_resource_log(self, el):
        # construct reversed dict: C -> CT
        rev_ctypes = dict()
        for type_name, type_value in self._ctypes.items():
            for v in type_value:
                rev_ctypes[v] = type_name

        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': rev_ctypes[event.case_id],
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
        # designate the partitioning (on C)
        # (map the values of a specified attribute to case_ids)
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

        # validate the partitioning
        is_disjoint = set.intersection(*par) == set()
        is_union = set.union(*par) == set(el['case_id'])

        # build types
        if is_disjoint and is_union:
            for i, values in enumerate(par):
                self._ctypes['CT.{}'.format(i)] = values.copy()
            self.is_ctypes_verified = True
        else:
            self.is_ctypes_verified = False

    def derive_resource_log(self, el):
        # construct reversed dict: A -> AT
        rev_atypes = dict()
        for type_name, type_value in self._atypes.items():
            for v in type_value:
                rev_atypes[v] = type_name
        # construct reversed dict: C -> CT
        rev_ctypes = dict()
        for type_name, type_value in self._ctypes.items():
            for v in type_value:
                rev_ctypes[v] = type_name

        # iterate through all events in the original log and convert
        # Note: only E_res (resource events) should be considered
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': rev_ctypes[event.case_id],
                'activity_type': rev_atypes[event.activity],
                'time_type': ''
            })

        from pandas import DataFrame
        return DataFrame(rl)

