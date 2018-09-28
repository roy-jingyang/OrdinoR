# -*- coding: utf-8 -*-

'''
This module contains the definition of a class which is a implementation of the
baseline execution model mining approach, i.e.
    AT = {{a} | a in A}, CT = {C} (not considered), TT = {T} (not considered).
'''

from .base import BaseExecutionModeMiner

class NaiveActivityNameExecutionModeMiner(BaseExecutionModeMiner):
    
    def __init__(self, el):
        # ignore the case type dimension
        self._build_atypes(el)
        # ignore the time type dimension
        self.verify()

    def _build_atypes(self, el):
        activity_names = [{x} for x in 
                sorted(el.groupby('activity').groups.keys())]
        #is_disjoint = set.intersection(*activity_names)
        is_disjoint = True # naturally
        #is_union = set.union(*activity_names)
        is_union = True # naturally

        if is_disjoint and is_union:
            self.is_atypes_verified = True
            for i, coll in enumerate(activity_names):
                self._atypes['AT.{}'.format(i)] = coll.copy()
        else:
            self.is_atypes_verified = False

    def derive_resource_log(self, el):
        # construct reversed dict: A -> AT
        rev_atypes = dict()
        for type_name, identifiers in self._atypes.items():
            for i in identifiers:
                rev_atypes[i] = type_name

        # iterate through all events in the original log and convert
        resource_log = list()
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': None,
                'activity_type': rev_atypes[event.activity],
                'time_type': None
            })

        from pandas import DataFrame
        return DataFrame(rl)

