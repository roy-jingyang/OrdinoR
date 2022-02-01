"""
Definition of rule:

A conjunction of atomic rules.
"""

from ordinor.utils.validation import check_convert_input_log
from .AtomicRule import AtomicRule

class Rule(object):
    def __init__(self, ars):
        """
        Parameters
        ----------
        ars : list of AtomicRule
            A list of atomic rules.
        """
        self.ars = ars
    
    @property
    def is_null(self) -> bool:
        return len(self.ars) == 1 and self.ars[0].is_null
    
    def __repr__(self) -> str:
        if self.is_null:
            return str(self.ars[0])
        else:
            return ' ∧ '.join(
                f'({ar})' for ar in 
                sorted(self.ars, key=lambda _ar: _ar.attr)
                if not ar.is_null
            )
    
    def get_attrs(self) -> set:
        return set(sorted(ar.attr for ar in self.ars if not ar.is_null))
    
    def __len__(self) -> int:
        return len([ar for ar in self.ars if not ar.is_null])
    
    def to_types(self) -> tuple: 
        ars_ct = []
        ars_at = []
        ars_tt = []
        for ar in self.ars:
            if not ar.is_null:
                if ar.attr_dim == 'CT':
                    ars_ct.append(ar)
                if ar.attr_dim == 'AT':
                    ars_at.append(ar)
                if ar.attr_dim == 'TT':
                    ars_tt.append(ar)
        rule_ct = Rule(ars_ct) if len(ars_ct) > 0 else Rule([AtomicRule()])
        rule_at = Rule(ars_at) if len(ars_at) > 0 else Rule([AtomicRule()])
        rule_tt = Rule(ars_tt) if len(ars_tt) > 0 else Rule([AtomicRule()])
        return rule_ct, rule_at, rule_tt

    def append(self, new_ar: AtomicRule):
        ars_same_attr = [
            ar for ar in self.ars
            if new_ar.is_same_attr(ar)
        ]
        if len(ars_same_attr) == 0:
            self.ars.append(new_ar)
        else:
            idx_stronger_ars = []
            idx_weaker_ars = []
            for i, ar in enumerate(self.ars):
                if ar >= new_ar:
                    idx_stronger_ars.append(i)
                if ar < new_ar:
                    idx_weaker_ars.append(i)
            if len(idx_stronger_ars) > 0:
                # ignore if there is any stronger atomic rule
                pass
            else:
                if len(idx_weaker_ars) > 0:
                    # remove any weaker atomic rule before appending
                    for i in idx_weaker_ars:
                        del self.ars[i]
                # append the new atomic rule
                self.ars.append(new_ar)
    
    def apply(self, el, index_only=False):
        """
        Apply a rule to an event log, selecting rows which meet the
        criteria defined by the conjunction of the atomic rules.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the rule will be applied.
        index_only : bool, optional
            Whether to return only the indices of rows. Defaults to `False`.

        Returns
        -------
        sublog : pandas.DataFrame
            A subset of the input event log after applying the rule.
        """
        sublog = check_convert_input_log(el)
        for ar in self.ars:
            sublog = ar.apply(sublog, index_only=False)
        if index_only:
            return sublog.index
        else:
            return sublog
