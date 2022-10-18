"""
Calculate quality measures using internal representation instead of a set
of execution contexts and an event log. 

This calculation is designed to be efficient, so that iterative learning
procedures guided by dispersal and impurity can be boosted. As such, the
expected inputs are data structures that capture the internal
representation of a derived resource log. 
"""

from math import comb
from itertools import combinations

from scipy.stats import entropy
from scipy.spatial.distance import hamming
import pandas as pd

def impurity(m_event_co, m_event_r):
    """
    Calculate impurity by considering resource ids as ground-truth and
    execution context labels as predicted labels.

    Parameters
    ----------
    m_event_co : pandas.Series
        An array indexed by event ids, recording labels of the execution
        contexts to which the events belong to.
    
    m_event_r : pandas.Series
        An array indexed by event ids, recording ids of the resources who
        originated the events.
    
    Returns
    -------
    total_impurity : float
        The result impurity.

    Notes
    -----
    * All events are expected to be events with resource information,
      i.e., `m_event_co` and `m_event_r` should have equal lengths.
    * The event index is expected to be consistent, i.e., `m_event_co`
      and `m_event_r` should share the same index.
    """
    if len(m_event_co) != len(m_event_r):
        raise ValueError('Array lengths unmatched')
    N_events = len(m_event_r)

    df = pd.DataFrame({'_co': m_event_co, '_r': m_event_r})

    # entropy-based

    total_entropy = 0
    for co, rows in df.groupby('_co'):
        wt = len(rows) / N_events
        # calculate the discrete distribution
        pk = rows['_r'].value_counts(normalize=True)
        if len(pk) > 1:
            # entropy
            total_entropy += wt * entropy(pk, base=2)
        else:
            total_entropy += 0
    
    # standardize
    max_entropy = entropy(
        df['_r'].value_counts(normalize=True),
        base=2
    )
    return total_entropy / max_entropy

    '''
    # misclassification-error-based

    total_error = 0
    for co, rows in df.groupby('_co'):
        wt = len(rows) / N_events
        # calculate the discrete distribution
        pk = rows['_r'].value_counts(normalize=True)
        if len(pk) > 1:
            # entropy
            total_error += wt * (1 - max(pk))
        else:
            total_error += 0

    # standardize
    max_error = 1 - max(df['_r'].value_counts(normalize=True))
    return total_error / max_error

    return total_error
    '''

def dispersal(m_co_t, m_event_co, m_event_r):
    """
    Calculate dispersal based on event pairwise distance, using
    "execution context pairwise distance" as proxy for improved
    computational efficiency.

    Parameters
    ----------
    m_co_t : pandas.DataFrame
        An array indexed by execution context ids, recording labels of
        the case types, activity types, and time types of execution
        contexts, i.e., the column number is 3.

    m_event_co : pandas.Series
        An array indexed by event ids, recording labels of the execution
        contexts to which the events belong to.
    
    m_event_r : pandas.Series
        An array indexed by event ids, recording ids of the resources who
        originated the events.

    Returns
    -------
    total_dispersal : float
        The result dispersal.

    Notes
    -----
    * All events are expected to be events with resource information.
    * The execution context index is expected to be consistent with the
      labels recorded for events, i.e., the index of `m_co_t` should be
      the same as the unique values contained in `m_event_co`.
    """
    if len(m_event_co) != len(m_event_r):
        raise ValueError('Array lengths unmatched')
    N_events = len(m_event_r)
    N_dims = sum(m_co_t.sum(axis=0) > 0)
    
    if N_dims == 0:
        return 0.0
    
    data = {'_co': m_event_co, '_r': m_event_r}
    df = pd.DataFrame(data)
    total_dispersal = 0.0
    for r, rows in df.groupby('_r'):
        n_rows = len(rows)
        count_co_events = rows['_co'].value_counts()
        wt =  n_rows / N_events
        # find all involved execution contexts
        co_ids = m_event_co.loc[rows.index].unique()
        if len(co_ids) < 2:
            avg_event_pdist = 0
        else:
            sum_event_pdist_across = 0
            # enumerate to calculate sum of pairwise distance (across)
            # NOTE: scipy's hamming dist. is standardized based on # dims
            for na, nb in combinations(co_ids, r=2):
                dist_na_nb = hamming(
                    m_co_t.loc[na].to_numpy(),
                    m_co_t.loc[nb].to_numpy()
                ) * (3 / N_dims)
                N_events_na = count_co_events.loc[na]
                N_events_nb = count_co_events.loc[nb]
                sum_event_pdist_across += (
                    dist_na_nb * N_events_na * N_events_nb
                )
            # total event pairwise distance (across + within; within are 0s)
            sum_event_pdist = sum_event_pdist_across + 0
            n_total_pairs = comb(n_rows, 2)
            avg_event_pdist = sum_event_pdist / n_total_pairs

        total_dispersal += wt * avg_event_pdist

    return total_dispersal