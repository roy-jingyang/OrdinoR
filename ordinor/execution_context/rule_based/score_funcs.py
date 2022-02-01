"""
Calculate quality measures using internal representation instead of a set
of execution contexts and an event log. 

This calculation is designed to be more efficient, so that iterative
learning procedures guided by dispersal and impurity can be boosted.
"""

from scipy.stats import entropy
from scipy.spatial.distance import pdist
import pandas as pd

def impurity(m_event_co, m_event_r):
    """
    Calculate impurity by considering resource ids as groundtruth and
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
    total_impurity = 0
    for co, rows in df.groupby('_co'):
        wt = len(rows) / N_events
        # calculate the discrete distribution
        pk = rows['_r'].value_counts(normalize=True)
        if len(pk) > 1:
            # entropy
            total_impurity += wt * entropy(pk, base=2)
        else:
            total_impurity += 0
    return total_impurity

def dispersal(m_event_ct, m_event_at, m_event_tt, m_event_r):
    """

    Parameters
    ----------
    m_event_ct : pandas.DataFrame
        An array indexed by event ids, recording labels of the case types
        of events.

    m_event_at : pandas.DataFrame
        An array indexed by event ids, recording labels of the activity
        types of events.

    m_event_tt : pandas.DataFrame
        An array indexed by event ids, recording labels of the time types
        of events.
    
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
    * Any of `m_event_ct`, `m_event_at`, and `m_event_tt` can be None,
      indicating that specific process dimension is not considered.
    * The event index is expected to be consistent, i.e., `m_event_ct`,
      `m_event_at`, `m_event_tt`, and `m_event_r` should share the same
      index.
    """
    dims = []
    data = {'_ct': m_event_ct, '_at': m_event_at, '_tt': m_event_tt}
    for t in ['_ct', '_at', '_tt']:
        if data[t] is not None:
            if len(data[t]) != len(m_event_r):
                raise ValueError('Array lengths unmatched')
            dims.append(t)
        else:
            del data[t]
    N_events = len(m_event_r)
    data['_r'] = m_event_r

    df = pd.DataFrame(data)
    total_dispersal = 0
    for r, rows in df.groupby('_r'):
        n_rows = len(rows)
        wt =  n_rows / N_events
        if n_rows > 1:
            # 2 or more events
            avg_event_pdist = pdist(
                df[dims].to_numpy(), metric='hamming'
            )
            avg_event_pdist = avg_event_pdist.mean()
        else:
            # only 1 event
            avg_event_pdist = 0
        
        total_dispersal += wt * avg_event_pdist
    return total_dispersal
