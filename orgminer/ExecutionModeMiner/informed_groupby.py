# -*- coding: utf-8 -*-

'''
This module contains the implementations of execution mode learning methods
that derive C/A/T types based on externally provided information other than the
log data that informs the subsequent mappings of C/A/T:
    - TraceClusteringCTMiner
'''

from .direct_groupby import CTonlyMiner, ATTTMiner

class TraceClusteringCTMiner(CTonlyMiner):
    '''(CT only method) Let each of the trace clusters be a case type and the 
    corresponding (identifiers of) cases be mapped onto the types.
    '''

    def __init__(self, el, fn_partition):
        self._build_ctypes(el, fn_partition)
        self._build_atypes(el)
        self._build_ttypes(el)
        self.verify()

    def _build_ctypes(self, el, fn_partition):
        self._ctypes = dict()
        par = list()
        with open(fn_partition, 'r') as f_par:
            for line in f_par:
                case_id, cluster = (line.split('\t')[0].strip(), 
                    line.split('\t')[1].strip())
                self._ctypes[case_id] = 'CT.{}'.format(cluster)

        self.is_ctypes_verified = self.verify_partition(
            set(el['case_id']), self._ctypes)

# TODO: better ways to handle multiple inheritance?
class TraceClusteringFullMiner(TraceClusteringCTMiner, ATTTMiner):
    '''(CT + AT + TT method) Based on TraceClusteringCTMiner (CT only) and
    ATTTMiner (AT + TT). All three dimensions are considered.
    '''
    def __init__(self, el, 
        fn_partition, resolution, datetime_format='%Y/%m/%d %H:%M:%S.%f'):
        TraceClusteringCTMiner._build_ctypes(self, el, fn_partition)
        ATTTMiner._build_atypes(self, el)
        ATTTMiner._build_ttypes(self, el, resolution, datetime_format)
        self.verify()

    def derive_resource_log(self, el):
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

