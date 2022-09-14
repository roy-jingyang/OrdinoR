"""
Measure the quality of a set of execution contexts with respect to an
event log. Expected input is a derived resource(-event) log, i.e., a
resource(-event) log derived from a given event log and a set of
execution contexts.

This implementation utilizes the efficient calculation approach devised
for rule-based execution context learning. A derived resource log is
transformed into three specific data structures, which are then fed to
the calculation.
"""

import pandas as pd
import ordinor.constants as const

from .rule_based import impurity as _impurity
from .rule_based import dispersal as _dispersal

def impurity(rl):
    """
    Calculate impurity by considering resource ids as ground-truth and
    execution context labels as predicted labels.

    Parameters
    ----------
    rl : pandas.DataFrame
        A derived resource log as a pandas DataFrame.

    Returns
    -------
    total_impurity : float
        The result impurity.
    """
    COLS = [const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE]
    # m_event_co : pandas.Series
    #     An array indexed by event ids, recording labels of the execution
    #     contexts to which the events belong to.
    mat_event_co = rl[COLS]
    for col in COLS:
        codes, _ = pd.factorize(mat_event_co[col])
        mat_event_co.loc[:, col] = codes.astype(str)
    mat_event_co.loc[:, '_co'] = mat_event_co[COLS].agg('-'.join, axis=1)
    mat_event_co.loc[:, '_co'], _ = pd.factorize(mat_event_co['_co'])
    m_event_co = mat_event_co['_co']
    # m_event_r : pandas.Series
    #     An array indexed by event ids, recording ids of the resources who
    #     originated the events.
    m_event_r = rl[const.RESOURCE]
    return _impurity(
        m_event_co=m_event_co,
        m_event_r=m_event_r
    )

def dispersal(rl):
    """
    Calculate dispersal based on event pairwise distance, using
    "execution context pairwise distance" as proxy for improved
    computational efficiency.

    Parameters
    ----------
    rl : pandas.DataFrame
        A derived resource log as a pandas DataFrame.

    Returns
    -------
    total_dispersal : float
        The result dispersal.
    """
    COLS = [const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE]
    # m_event_co : pandas.Series
    #     An array indexed by event ids, recording labels of the execution
    #     contexts to which the events belong to.
    mat_event_co = rl[COLS]
    for col in COLS:
        codes, _ = pd.factorize(mat_event_co[col])
        mat_event_co.loc[:, col] = codes.astype(str)
    mat_event_co.loc[:, '_co'] = mat_event_co[COLS].agg('-'.join, axis=1)
    mat_event_co.loc[:, '_co'], _ = pd.factorize(mat_event_co['_co'])
    m_event_co = mat_event_co['_co']
    # m_co_t : pandas.DataFrame
    #     An array indexed by execution context ids, recording labels of
    #     the case types, activity types, and time types of execution
    #     contexts, i.e., the column number is 3.
    m_co_t = mat_event_co.drop_duplicates(subset='_co')
    m_co_t = m_co_t.set_index('_co')
    m_co_t = m_co_t.astype(int)
    # m_event_r : pandas.Series
    #     An array indexed by event ids, recording ids of the resources who
    #     originated the events.
    m_event_r = rl[const.RESOURCE]
    return _dispersal(
        m_co_t=m_co_t,
        m_event_co=m_event_co,
        m_event_r=m_event_r
    )
