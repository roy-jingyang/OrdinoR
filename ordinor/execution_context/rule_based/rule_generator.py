"""
Generating rules for a given event attribute based on a given event log.

Event attributes are considered generic attributes. Hence, there are two
types of generating functions which are different depending on whether an
attribute is numeric or categorical. 

"""

import numpy as np
import pandas as pd

from ordinor.utils.validation import check_convert_input_log
from ordinor.utils.set_utils import unique_k_partitions

from .AtomicRule import AtomicRule
from .Rule import Rule

class NumericRuleGenerator:
    @classmethod
    def HistogramSplit(cls, attr, attr_dim, el, bins=10):
        """
        Generate rules for a given numeric attribute using histogram
        split on the attribute values in the input event log.

        Parameters
        ----------
        attr : str
            The name of an event attribute.
        attr_dim: str
            The process dimension of an event attribute, denoted by one
            of the types. Can be one of {'CT', 'AT', 'TT'}.
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the atomic rule will be applied.
        bins : int or sequence of scalars or str, optional
            Defaults to `10`, i.e., rules corresponded to 10 equal-width
            bins will be generated.
            See numpy.histogram_bin_edges for more explanation.

        Returns
        -------
        l_rules : list of Rule
            A list of rules generated for the split.
        
        See Also
        --------
        numpy.histogram_bin_edges : 
            Function to calculate only the edges of the bins used by the
            histogram function.
        """
        el = check_convert_input_log(el)
        rules = []

        arr = el[attr]
        hist_bin_edges = np.histogram_bin_edges(arr, bins=bins)
        n_bins = len(hist_bin_edges) - 1
        for i in range(len(hist_bin_edges)):
            if i > 0:
                left_edge = hist_bin_edges[i-1]
                right_edge = hist_bin_edges[i]
                closed = 'left' if i < n_bins else 'both'
                ar = AtomicRule(
                    attr=attr, attr_type='numeric', 
                    attr_vals=pd.Interval(left_edge, right_edge, closed=closed),
                    attr_dim=attr_dim
                )
                rules.append(Rule(ars=[ar]))
        return rules

class CategoricalRuleGenerator:
    @classmethod
    def RandomTwoWayPartition(cls, attr, attr_dim, el):
        """
        Generate rules for a given numeric attribute using histogram
        split on the attribute values in the input event log.

        Parameters
        ----------
        attr : str
            The name of an event attribute.
        attr_dim: str
            The process dimension of an event attribute, denoted by one
            of the types. Can be one of {'CT', 'AT', 'TT'}.
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the atomic rule will be applied.

        Returns
        -------
        generator
            List of rules generated for the two way partitioning.
        
        Notes
        -----
        * The total possible number two way partitioning on a size-N set
          (note that empty sets are not included) is `2^(N-1) - 1`, i.e.,
          `(2^N - 2) / 2` 
        """
        unique_attr_vals = set(el[attr])
        for pars in unique_k_partitions(unique_attr_vals, k=2):
            ar_left = AtomicRule(
                attr=attr, attr_type='categorical', 
                attr_vals=pars[0], attr_dim=attr_dim
            )
            ar_right = AtomicRule(
                attr=attr, attr_type='categorical', 
                attr_vals=pars[1], attr_dim=attr_dim
            )
            yield [Rule(ars=[ar_left]), Rule(ars=[ar_right])]
