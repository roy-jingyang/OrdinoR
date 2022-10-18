"""
Definition of rule:

A conjunction of atomic rules.
"""

from copy import deepcopy
import numpy as np

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
        all_null = True
        for ar in self.ars:
            if not ar.is_null:
                return False
        return all_null
    
    def __repr__(self) -> str:
        if self.is_null:
            return str(AtomicRule(attr=None))
        else:
            return ' \u2227 '.join(
                f'({ar})' for ar in 
                sorted(self.ars, key=lambda _ar: _ar.attr)
                if not ar.is_null
            )
    
    def get_attrs(self) -> set:
        return set(sorted(ar.attr for ar in self.ars if not ar.is_null))
    
    def __len__(self) -> int:
        return len([ar for ar in self.ars if not ar.is_null])
    
    def __hash__(self) -> int:
        tuple_ars = tuple(sorted(self.ars, key=lambda ar: ar.attr))
        return hash(tuple_ars)
    
    def __eq__(self, other) -> bool:
        if self.is_null and other.is_null:
            return True

        if len(self) != len(other):
            return False
        else:
            idx_other_ars = list(range(len(other)))
            has_eq_ar = []
            for ar in self.ars:
                has_eq = False
                for idx in idx_other_ars:
                    if other.ars[idx] == ar:
                        has_eq = True
                        break
                if has_eq:
                    idx_other_ars.remove(idx)
                has_eq_ar.append(has_eq)
            return np.all([has_eq_ar])
    
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
        idx_same_attr_ars = []
        for i, ar in enumerate(self.ars):
            if new_ar.is_same_attr(ar):
                idx_same_attr_ars.append(i)
        
        for idx in idx_same_attr_ars:
            existing_ar = self.ars[idx]
            if existing_ar >= new_ar:
                # if stronger rule exists, do nothing (return)
                return
            if existing_ar < new_ar:
                # remove any weaker atomic rule before appending
                self.ars.pop(idx)
        # append the new atomic rule
        self.ars.append(new_ar)
    
    def extend(self, new_r):
        for ar in new_r.ars:
            self.append(ar)
    
    def subrule(self, attr):
        l_same_attr_ars = []
        for i, ar in enumerate(self.ars):
            if ar.attr == attr:
                l_same_attr_ars.append(ar)
        return Rule(l_same_attr_ars)
    
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
        sublog = el
        # iteratively apply to the sublog
        for ar in self.ars:
            sublog = ar.apply(sublog, index_only=False)

        if index_only:
            return sublog.index
        else:
            return sublog
