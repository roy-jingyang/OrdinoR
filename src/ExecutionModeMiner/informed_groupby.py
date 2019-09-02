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
                case_id, cluster= line.split('\t')[0], line.split('\t')[1]
                par.append(


