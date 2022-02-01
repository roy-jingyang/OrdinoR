import ordinor.exceptions as exc
from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const

from ordinor.execution_context.base import BaseMiner

class TreeInductionMiner(BaseMiner):
    def __init__(self, el, specification):
        if el is not None:
            el = check_convert_input_log(el)
        # TODO
        pass

    def _build_ctypes(self, el, **kwargs):
        # TODO
        pass

    def _build_atypes(self, el, **kwargs):
        # TODO
        pass

    def _build_ttypes(self, el, **kwargs):
        # TODO
        pass
