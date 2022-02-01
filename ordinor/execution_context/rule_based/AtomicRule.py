"""
Definition of atomic rule:

A Boolean formula that concerns only a single event attribute.
`
    {\pi_{attr}(e)} \in {attr\_vals}
`
"""

import pandas as pd

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

# Boolean formula that concerns only a single event attribute:
# Pi_attr (e) \in attr_vals
class AtomicRule(object):
    def __init__(self, attr=None, attr_type=None, attr_vals=None, attr_dim=None):
        """
        Parameters
        ----------
        attr : str
            The name of an event attribute. If None, then a null
            AtomicRule is constructed.
        attr_type: str
            The data type of an event attribute. Can be one of
            {'numeric', 'categorical'}.
        attr_vals: set
            The attribute values of an event attribute. If the attribute
            is categorical, then a set of strings or numbers is expected;
            if the attribute is numeric, then a pandas.Interval (of
            numbers) is expected.
        attr_dim: str
            The process dimension of an event attribute, denoted by one
            of the types. Can be one of {'CT', 'AT', 'TT'}.
        """
        if attr is None or attr == '':
            self.attr = ''
        else:
            self.attr = attr
            if attr_type in {'numeric', 'categorical'}:
                self.attr_type = attr_type
            else:
                raise ValueError("`attr_type` must be one of {'numeric', 'categorical'}")
            if self.attr_type == 'numeric':
                if type(attr_vals) is pd.Interval:
                    self.attr_vals = attr_vals
                else:
                    raise ValueError("`attr_vals` must be of type `pd.Interval`")
            else:
                if type(attr_vals) is set:
                    self.attr_vals = attr_vals
                else:
                    raise ValueError("`attr_vals` must be of type `set`")
            if attr_dim in {'CT', 'AT', 'TT'}:
                self.attr_dim = attr_dim
            else:
                raise ValueError("`attr_dim` must be one of {'CT', 'AT', 'TT'}")
    
    @property
    def is_null(self) -> bool:
        return self.attr == ''
    
    def __repr__(self) -> str:
        if self.is_null:
            return '⊥ (null)'
        else:
            return f'`{self.attr}` ∈ {self.attr_vals}'

    def is_same_attr(self, other) -> bool:
        if self.is_null and other.is_null:
            return True
        else:
            return (
                self.attr == other.attr and
                self.attr_type == other.attr_type and
                self.attr_dim == other.attr_dim
            )
    
    def __eq__(self, other) -> bool:
        if self.is_null and other.is_null:
            return True
        elif self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # set equality
                return (
                    self.attr_vals >= other.attr_vals and 
                    self.attr_vals <= other.attr_vals
                )
            else:
                # interval equality
                return self.attr_vals == other.attr_vals
        else:
            return NotImplemented
    
    # null < (self) weaker rule < (other) stronger rule
    def __lt__(self, other) -> bool:
        if self.is_null:
            return not other.is_null
        if other.is_null:
            return False

        if self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # self is a superset
                return self.attr_vals > other.attr_vals
            else:
                # self has a larger interval that contains other's
                m, n = self.attr_vals.left, self.attr_vals.right
                a, b = other.attr_vals.left, other.attr_vals.right
                if m <= a and b <= n:
                    if m == a and b == n:
                        if self.attr_vals != other.attr_vals:
                            return (
                                self.attr_vals.closed == 'both' or 
                                other.attr_vals.closed == 'neither'
                            )
                    else:
                        return True
        else:
            return NotImplemented
    
    def __le__(self, other) -> bool:
        return self < other or self == other

    # (self) stronger rule > (other) weaker rule > null
    def __gt__(self, other) -> bool:
        if self.is_null:
            return False
        if other.is_null:
            return not self.is_null

        if self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # self is a subset
                return self.attr_vals < other.attr_vals
            else:
                # self has a smaller interval inside other's
                a, b = self.attr_vals.left, self.attr_vals.right
                m, n = other.attr_vals.left, other.attr_vals.right
                if m <= a and b <= n:
                    if m == a and b == n:
                        if self.attr_vals != other.attr_vals:
                            return (
                                self.attr_vals.closed == 'both' or 
                                other.attr_vals.closed == 'neither'
                            )
                    else:
                        return True
        else:
            return NotImplemented

    def __ge__(self, other) -> bool:
        return self > other or self == other
    
    def selector(self, el):
        """
        Generate from an atomic rule a boolean mask to select rows in an
        event log.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the atomic rule will be applied.

        Returns
        -------
        mask : pandas.Series
            Series (a boolean vector) representing the elements selected
            based on applying the atomic rule to the corresponding column.
        """
        el = check_convert_input_log(el)
        if self.is_null:
            mask = [True] * len(el)
        elif self.attr_type == 'categorical':
            mask = el[self.attr].isin(self.attr_vals)
        else:
            mask = el[self.attr].between(
                left=self.attr_vals.left,
                right=self.attr_vals.right,
                inclusive=self.attr_vals.closed
            )
        return mask
    
    def apply(self, el, index_only=False):
        """
        Apply an atomic rule to an event log, selecting rows which meet
        the criterion defined by the atomic rule.

        Parameters
        ----------
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the atomic rule will be applied.
        index_only : bool, optional
            Whether to return only the indices of rows. Defaults to `False`.

        Returns
        -------
        sublog : pandas.DataFrame
            A subset of the input event log after applying the rule.
        """
        el = check_convert_input_log(el)
        sublog = el[self.selector(el)]
        if index_only:
            return sublog.index
        else:
            return sublog
