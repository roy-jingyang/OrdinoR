"""
Generating rules for a given event attribute based on a given event log.

Event attributes are considered generic attributes. Hence, there are two
types of generating functions which are different depending on whether an
attribute is numeric or categorical. 

"""
from random import sample

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
    def RandomTwoSubsetPartition(cls, attr, attr_dim, el, n_sample=1, max_n_sample=100):
        """
        Generate rules for a given categorical attribute by performing a
        two-subset partitioning on all unique attribute values in the
        input event log.

        Parameters
        ----------
        attr : str
            The name of an event attribute.
        attr_dim: str
            The process dimension of an event attribute, denoted by one
            of the types. Can be one of {'CT', 'AT', 'TT'}.
        el : pandas.DataFrame, or pm4py EventLog
            An event log to which the atomic rule will be applied.
        
        n_sample : int or float, optional
            Sample size, must be a positive integer smaller or equal to
            the total possible number of partitions (see notes), or a
            positive float number smaller than 1.
            Defaults to integer `1`, i.e., sample will be of size 1.
            If not provided, this defaults to the value of
            `max_n_sample`.
        
        max_n_sample : int, optional
            Maximum sample size allowed, must be a positive integer
            smaller or equal to the total possible number of partitions.
            Defaults to `100`, i.e., sample size will be at most 100.

        Returns
        -------
        generator
            List of rules generated for the two-subset partitioning.
        
        Notes
        -----
        * The total possible number of two-subset partitions on a size-N
          set is `2^(N-1) - 1`, i.e., `(2^N - 2) / 2`. This is known as
          the Stirling number of the second kind, with k=2.
        * If `n_sample` is given as a valid float number, it is
          considered a percentage of the population.
        """
        el = check_convert_input_log(el)
        unique_attr_vals = set(el[attr])

        # calculate the number of all possibilities
        N_partitions = 2 ** (len(unique_attr_vals) - 1) - 1

        if n_sample is None:
            n_sample = max_n_sample
        elif type(n_sample) is int and n_sample > 0:
            pass
        elif type(n_sample) is float and 0 < n_sample and n_sample < 1:
            n_sample = int(N_partitions * n_sample)
            n_sample = 1 if n_sample < 1 else n_sample
        else:
            raise ValueError
        
        # cap sample size
        if n_sample > max_n_sample:
            n_sample = max_n_sample

        if n_sample >= N_partitions:
            # use entire population, if allowed
            indices = list(range(N_partitions))
        else:
            # otherwise, sample uniformly
            indices = sorted(sample(range(N_partitions), n_sample))
        
        j = 0
        for i, par in enumerate(unique_k_partitions(unique_attr_vals, k=2)):
            if j == len(indices):
                break
            
            if i == indices[j]:
                j += 1
                ar_left = AtomicRule(
                    attr=attr, attr_type='categorical', 
                    attr_vals=par[0], attr_dim=attr_dim
                )
                ar_right = AtomicRule(
                    attr=attr, attr_type='categorical', 
                    attr_vals=par[1], attr_dim=attr_dim
                )
                yield [Rule(ars=[ar_left]), Rule(ars=[ar_right])]
            else:
                pass
