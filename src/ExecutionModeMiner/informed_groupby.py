# -*- coding: utf-8 -*-

'''
This module contains the implementations of execution mode learning methods
that derive C/A/T types based on externally provided information other than the
log data that informs the subsequent mappings of C/A/T:
    - TraceClusteringCTMiner
'''

from .direct_groupby import ATTTMiner

class TraceClusteringCTMiner(ATTTMiner):
    '''Based on ATTTMiner (AT + TT), let each of the trace clusters be a case
    type and the corresponding (identifiers of) cases be mapped onto the types.
    '''

    def __init__(self, el, fn_partition):
        self._build_ctypes(el, fn_partition)
        self._build_atypes(el)
        self._build_ttypes(el)
        self.verify()

    def _build_ctypes(self, el, fn_partition):
        par = list()
        with open(fn_partition, 'r') as f_par:
            for line in f_par:
                case_id, cluster = line.split('\t')[0], line.split('\t')[1]
                self._ctypes[case_id] = 'CT.{}'.format(cluster)
        self._n_ctypes = len(set(self._ctypes.values()))

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
                    'time_type': self._ttypes[event.timestamp],
                })

        from pandas import DataFrame
        return DataFrame(rl)

