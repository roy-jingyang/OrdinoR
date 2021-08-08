"""
Execution context learning approaches that apply external tools ("proxy")
to derive information additional to the raw event log, namely

    - TraceClusteringCTMiner (CT only)
    - TraceClusteringFullMiner (CT & AT & TT)

"""

from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const

from .direct_attribute import CTonlyMiner, ATTTMiner

class TraceClusteringCTMiner(CTonlyMiner):
    """
    Informed by the result of applying trace clustering, each 
    variant of cases is taken as a case type.

    See Also
    --------
    ordinor.exe_context_miner.direct_attribute.CTonlyMiner
    """

    def __init__(self, el, fn_partition):
        self._build_ctypes(el, fn_partition)
        CTonlyMiner._build_atypes(self, el)
        CTonlyMiner._build_ttypes(self, el)
        self._verify()


    def _build_ctypes(self, el, fn_partition):
        el = check_convert_input_log(el)
        self._ctypes = dict()
        with open(fn_partition, 'r') as f_par:
            for line in f_par:
                case_id, cluster = (line.split('\t')[0].strip(), 
                    line.split('\t')[1].strip())
                self._ctypes[case_id] = 'CT.{}'.format(cluster)
        self.is_ctypes_verified = self._verify_partition(
            set(el[const.CASE_ID]), self._ctypes)


class TraceClusteringFullMiner(TraceClusteringCTMiner, ATTTMiner):
    """
    Informed by the result of applying trace clustering, each 
    variant of cases is taken as a case type, each value of activity 
    (task) label is taken as an activity type, and each possible value 
    of a designated datetime unit is taken as a time type.

    See Also
    --------
    ordinor.exe_context_miner.direct_attribute.FullMiner
    ordinor.exe_context_miner.proxy.TraceClusteringCTMiner
    """

    def __init__(self, el, 
        fn_partition, resolution):
        TraceClusteringCTMiner._build_ctypes(self, el, fn_partition)
        ATTTMiner._build_atypes(self, el)
        ATTTMiner._build_ttypes(self, el, resolution)
        self._verify()
