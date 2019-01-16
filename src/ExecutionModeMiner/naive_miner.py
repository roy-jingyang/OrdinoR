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
    an activity type, ignoring the case and time dimension.
    '''
    
    def _build_atypes(self, el):
        # designate the partitioning (on A)
        # (map each activity names to itself)
        par = [{a} for a in sorted(el['activity'].unique())]

        # validate the partitioning
        is_disjoint = True # naturally
        is_union = True # naturally

        # build types
        if is_disjoint and is_union:
            for i, values in enumerate(par):
                self._atypes['AT.{}'.format(i)] = values.copy()
            self.is_atypes_verified = True
        else:
            self.is_atypes_verified = False

    def _build_ctypes(self, el):
        # ignoring the case dimension: all events are of the same case type.
        self.is_ctypes_verified = True
        pass

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
        resource_log = list()
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
        for v, events in el.groupby(case_attr_name): # sorted by default
            par.append(set(events['case_id']))


        # validate the partitioning
        is_disjoint = True # naturally
        is_union = True # naturally

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
        resource_log = list()
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

