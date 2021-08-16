"""
Measures and methods for calculating precision
"""

import ordinor.exceptions as exc
import ordinor.constants as const

from .fitness import _is_conformed_event

def _is_allowed_event(event, om):
    m = (event[const.CASE_TYPE], 
         event[const.ACTIVITY_TYPE], 
         event[const.TIME_TYPE]
    )
    cand_groups = om.find_candidate_groups(m)
    return len(cand_groups) > 0


def solo_originator(rl, om):
    """
    Calculate precision by comparing a model against an ideal situation
    where each event should be allowed exactly one originator.

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.

    Returns
    -------
    float
        The resulting precision value.

    Notes
    -----
    A resource log, instead of an event log, is used here, hence only
    events with resource information in the original event log are
    considered.
    """
    cand_E = set()

    for event in rl.to_dict(orient='records'):
        if _is_allowed_event(event, om):
            m = (event[const.CASE_TYPE], 
                 event[const.ACTIVITY_TYPE], 
                 event[const.TIME_TYPE]
            )
            cand_groups = om.find_candidate_groups(m)

            cand_e = frozenset.union(*cand_groups)
            cand_E.update(cand_e)
    n_cand_E = len(cand_E)

    if n_cand_E == 0:
        exc.warn_nan_returned(
            'Precision is undefined with zero candidate resource.'
        )
        return float('nan')
    else:
        n_allowed_events = 0
        precision = 0.0

        for event in rl.to_dict(orient='records'):
            if _is_allowed_event(event, om):
                n_allowed_events += 1

                m = (event[const.CASE_TYPE], 
                    event[const.ACTIVITY_TYPE], 
                    event[const.TIME_TYPE]
                )
                cand_groups = om.find_candidate_groups(m)
                cand_e = frozenset.union(*cand_groups)
                n_cand_e = len(cand_e)

                if _is_conformed_event(event, om):
                    # give reward
                    precision += (n_cand_E + 1 - n_cand_e) / n_cand_E
                else:
                    # NOTE: no extra penalty is given
                    precision += 0.0

        precision *= 1 / n_allowed_events
        return precision


# "rc-measure"
def solo_originator_flower_zero(rl, om):
    """
    Calculate precision by comparing a model against an ideal
    situation where each event should be allowed exactly one originator.
    
    Only conformed events are considered.

    "Flower" models (ones that fit all possible behavior) are considered
    having zero precision.

    Parameters
    ----------
    rl : pandas.DataFrame
        A resource log.
    om : OrganizationalModel
        An organizational model.

    Returns
    -------
    float
        The resulting precision value.

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
    
    # list of "cand(e)"
    l_n_cand_e = list()
    cand_E = set()
    for event in conformed_events.to_dict(orient='records'):
        m = (event[const.CASE_TYPE], 
             event[const.ACTIVITY_TYPE], 
             event[const.TIME_TYPE]
        )
        cand_groups = om.find_candidate_groups(m)

        cand_e = frozenset.union(*cand_groups)
        l_n_cand_e.append(len(cand_e))
        cand_E.update(cand_e)

    # "|cand(E)|"
    n_cand_E = len(cand_E)

    if n_cand_E == 0:
        exc.warn_nan_returned(
            'Precision is undefined with zero candidate resource.'
        )
        return float('nan')
    elif n_cand_E == 1:
        exc.warn_runtime(
            'Assume precision=1.0 with only one candidate in total.'
        )
        return 1.0
    else:
        rc_sum = sum([
            (n_cand_E - n_cand_e) / (n_cand_E - 1) 
            for n_cand_e in l_n_cand_e
        ])
        return rc_sum / n_conformed_events


def conf_events_model_proportion(rl, om):
    """
    Calculate precision as the proportion of resource events in a "model"
    that are conformed, ignoring multiple occurrences.

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
    # count of all possible resource events allowed by the model
    n_allowed_res_events = sum(len(om.find_execution_contexts(r))
        for r in om.resources)
    return n_conformed_res_events / n_allowed_res_events


# precision (by resource events, counting frequencies)
def conf_events_model_proportion_freq(rl, om):
    """
    Calculate precision as the proportion of resource events in a "model"
    that are conformed, considering multiple occurrences (frequencies).

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

    # RE_conf
    conformed_res_events = set(
        (re[const.RESOURCE], 
         re[const.CASE_TYPE], 
         re[const.ACTIVITY_TYPE],
         re[const.TIME_TYPE]
        )
        for re in conformed_events.drop_duplicates().to_dict(orient='records')
    )

    ctx_occurrences = rl.groupby([
        const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE]).size().to_dict()

    F_conformed_res_events = 0.0
    F_allowed_res_events = 0.0
    F_model_false_res_events = 0.0
    for r in om.resources:
        for m in om.find_execution_contexts(r):
            res_event = (r, m[0], m[1], m[2])
            if res_event in conformed_res_events:
                F_conformed_res_events += ctx_occurrences[m]
            else:
                F_model_false_res_events += ctx_occurrences[m]
            F_allowed_res_events += ctx_occurrences[m]

    return F_conformed_res_events / F_allowed_res_events
