"""
Measures and methods for calculating fitness
"""

import ordinor.constants as const

def _is_conformed_event(event, om):
    m = (event[const.CASE_TYPE], 
         event[const.ACTIVITY_TYPE], 
         event[const.TIME_TYPE]
    )
    cand_groups = om.find_candidate_groups(m)
    for g in cand_groups:
        if event[const.RESOURCE] in g:
            return True
    return False


def conf_events_proportion(rl, om):
    """
    Calculate fitness as the proportion of events that are conformed.

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.

    Returns
    -------
    float
        The resulting fitness value.

    Notes
    -----
    A resource log, instead of an event log, is used here, hence only
    events with resource information in the original event log are
    considered.
    """
    conformed_events = rl[
        rl.apply(lambda e: _is_conformed_event(e, om), axis=1)
    ]
    # "|E_conf|"
    n_conformed_events = len(conformed_events) 
    # "|E_res|"
    n_events = len(rl) 
    return n_conformed_events / n_events


def conf_res_events_proportion(rl, om):
    """
    Calculate fitness as the proportion of resource events in a log that
    are conformed, ignoring multiple occurrences.

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.

    Returns
    -------
    float
        The resulting fitness value.

    Notes
    -----
    A resource log, instead of an event log, is used here, hence only
    events with resource information in the original event log are
    considered.
    Multiple occurrences (duplicates) of resources events are ignored. 
    """
    conformed_events = rl[
        rl.apply(lambda e: _is_conformed_event(e, om), axis=1)
    ]
    # "|RE_conf|"
    n_conformed_res_events = len(conformed_events.drop_duplicates())
    # "|RE|"
    n_actual_res_events = len(rl.drop_duplicates()) 
    return n_conformed_res_events / n_actual_res_events
