"""
Definition of atomic rule:

A Boolean formula that concerns only a single event attribute.
`
    {\pi_{attr}(e)} \in {attr\_vals}
`
"""
import pandas as pd

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
            {'numeric', 'categorical', 'boolean'}.
        attr_vals: set, pandas.Interval, or bool
            The attribute values of an event attribute. If the attribute
            is categorical, then a frozenset of strings or numbers is
            expected;
            if the attribute is numeric, then a pandas.Interval (of
            numbers) is expected; if the attribute is boolean, then a
            bool value is expected. 
        attr_dim: str
            The process dimension of an event attribute, denoted by one
            of the types. Can be one of {'CT', 'AT', 'TT'}.
        """
        if attr is None or attr == '':
            self.attr = None
            self.attr_type = None
            self.attr_vals = None
            self.attr_dim = None
        else:
            self.attr = attr
            if attr_type in {'numeric', 'categorical', 'boolean'}:
                self.attr_type = attr_type
            else:
                raise ValueError("`attr_type` must be one of {'numeric', 'categorical', 'boolean'}")
            if self.attr_type == 'numeric':
                if type(attr_vals) is pd.Interval:
                    self.attr_vals = attr_vals
                else:
                    raise ValueError("`attr_vals` must be of type `pd.Interval`")
            elif self.attr_type == 'categorical':
                if type(attr_vals) is frozenset:
                    self.attr_vals = attr_vals
                else:
                    raise ValueError("`attr_vals` must be of type `frozenset`")
            else:
                if type(attr_vals) is bool:
                    self.attr_vals = attr_vals
                else:
                    raise ValueError("`attr_vals` must be of type `bool`")

            if attr_dim in {'CT', 'AT', 'TT'}:
                self.attr_dim = attr_dim
            else:
                raise ValueError("`attr_dim` must be one of {'CT', 'AT', 'TT'}")
    
    @property
    def is_null(self) -> bool:
        return self.attr is None
    
    def __repr__(self) -> str:
        if self.is_null:
            return '\u22a5 (null)'
        elif self.attr_type == 'categorical':
            # categorical: show sorted set elements 
            return f'`{self.attr}` \u2208 ' + '{' + str(sorted(self.attr_vals))[1:-1] + '}'
        elif self.attr_type == 'numeric':
            # numeric: show intervals
            return f'`{self.attr}` \u2208 {self.attr_vals}'
        else:
            # boolean: show formula directly
            return f'`{self.attr}` == {self.attr_vals}'

    def is_same_attr(self, other) -> bool:
        if self.is_null and other.is_null:
            return True
        else:
            return (
                self.attr == other.attr and
                self.attr_type == other.attr_type and
                self.attr_dim == other.attr_dim
            )
        
    def __hash__(self) -> int:
        if self.is_null:
            return 0
        else:
            return hash(tuple((self.attr, self.attr_vals)))
    
    def __eq__(self, other) -> bool:
        if self.is_null and other.is_null:
            return True
        elif self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # categorical: set equality
                return (
                    self.attr_vals >= other.attr_vals and 
                    self.attr_vals <= other.attr_vals
                )
            else:
                # numeric: interval equality
                # boolean: truth value equality
                return self.attr_vals == other.attr_vals
        else:
            return False
    
    # null < (self) weaker rule < (other) stronger rule
    def __lt__(self, other) -> bool:
        if self.is_null:
            return not other.is_null
        if other.is_null:
            return False

        if self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # categorical: self is a superset
                return self.attr_vals > other.attr_vals
            elif self.attr_type == 'numeric':
                # numeric: self has a larger interval that contains other's
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
                # boolean: not implemented
                return NotImplemented
        else:
            return NotImplemented
    
    def __le__(self, other) -> bool:
        return self == other or self < other

    # (self) stronger rule > (other) weaker rule > null
    def __gt__(self, other) -> bool:
        if self.is_null:
            return False
        if other.is_null:
            return not self.is_null

        if self.is_same_attr(other):
            if self.attr_type == 'categorical':
                # categorical: self is a subset
                return self.attr_vals < other.attr_vals
            elif self.attr_type == 'numeric':
                # numeric: self has a smaller interval inside other's
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
                # boolean: not implemented
                return NotImplemented
        else:
            return NotImplemented

    def __ge__(self, other) -> bool:
        return self == other or self > other
    
    def __neg__(self):
        if self.attr_type == 'boolean':
            return AtomicRule(attr=self.attr, attr_type=self.attr_type, attr_vals=(not self.attr_vals), attr_dim=self.attr_dim)
        else:
            return NotImplemented
    
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
        if self.is_null:
            sublog = el
        elif self.attr_type == 'categorical':
            # categorical: .isin()
            sublog = el[el[self.attr].isin(self.attr_vals)]
        elif self.attr_type == 'numeric':
            # numeric: Series.between()
            sublog = el[
                el[self.attr].between(
                    left=self.attr_vals.left,
                    right=self.attr_vals.right,
                    inclusive=self.attr_vals.closed
                )
            ]
        else:
            # boolean: direct evaluation
            sublog = el[el[self.attr] == self.attr_vals]

        if index_only:
            return sublog.index
        else:
            return sublog
