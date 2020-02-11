# -*- coding: utf-8 -*-

"""This module contains the implementation of the global conformance 
checking measures.
"""
from deprecated import deprecated
from warnings import warn

def _is_conformed_event(event, om):
    m = (event.case_type, event.activity_type, event.time_type)
    cand_groups = om.find_candidate_groups(m)
    for g in cand_groups:
        if event.resource in g:
            return True
    return False


def _is_allowed_event(event, om):
    m = (event.case_type, event.activity_type, event.time_type)
    cand_groups = om.find_candidate_groups(m)
    return len(cand_groups) > 0


def fitness(rl, om):
    """Calculate the fitness of an organizational model against a given
    resource log.

    Parameters
    ----------
    rl : DataFrame
        A resource log.
    om : OrganizationalModel
        The discovered organizational model.

    Returns
    -------
    float
        The result fitness value.

    Notes
    -----
    A resource log instead of an event log is used here, however this 
    is valid since a resource log is (implicitly) one-to-one 
    corresponded with an event log.
    """
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events) # "|E_conf|"
    n_events = len(rl) # "|E_res|"
    return n_conformed_events / n_events


@deprecated(reason='This definition is neither being nor intended to be used.')
# fitness (by resource events)
def fitness_re(rl, om):
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    # "|RE_conf|"
    n_conformed_res_events = len(conformed_events.drop_duplicates())
    n_actual_res_events = len(rl.drop_duplicates()) # "|RE|"
    return n_conformed_res_events / n_actual_res_events


@deprecated(reason='This definition is not being used.')
# Wil's precision
def rc_measure(rl, om):
    """Calculate the precision of an organizational model against a 
    given event log. 
    
    Calculate model precision considering only "fitting" (conformed) 
    events.

    Parameters
    ----------
    rl : DataFrame
        A resource log.
    om : OrganizationalModel
        The discovered organizational model.

    Returns
    -------
    float
        The result precision value.

    Notes
    -----
    A resource log instead of an event log is used here, however this 
    is valid since a resource log is (implicitly) one-to-one 
    corresponded with an event log.
    """
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events) # "|E_conf|"
    
    l_n_cand_e = list() # list of "cand(e)"
    cand_E = set()
    for event in conformed_events.itertuples():
        m = (event.case_type, event.activity_type, event.time_type)
        cand_groups = om.find_candidate_groups(m)

        cand_e = frozenset.union(*cand_groups) # cand(e)
        l_n_cand_e.append(len(cand_e)) # "|cand(e)|"
        cand_E.update(cand_e)

    n_cand_E = len(cand_E) # "|cand(E)|"

    if n_cand_E == 0:
        warn('No candidate resource.', RuntimeWarning)
        return float('nan')
    elif n_cand_E == 1:
        warn('Overall number of candidate is 1.', RuntimeWarning)
        return 1.0
    else:
        rc_sum = sum([(n_cand_E - n_cand_e) / (n_cand_E - 1) 
            for n_cand_e in l_n_cand_e])
        return rc_sum / n_conformed_events


@deprecated(reason='This definition is not being used.')
# precision (by resource events, not counting frequencies)
def precision_re_nofreq(rl, om):
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    # "|RE_conf|"
    n_conformed_res_events = len(conformed_events.drop_duplicates())
    # count of all possible resource events allowed by the model
    n_allowed_res_events = sum(len(om.find_execution_modes(r))
        for r in om.resources)
    return n_conformed_res_events / n_allowed_res_events


@deprecated(reason='This definition is not being used.')
# precision (by resource events, counting frequencies)
def precision_re_freq(rl, om):
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events)

    # RE_conf
    conformed_res_events = set(
        (re.resource, re.case_type, re.activity_type, re.time_type)
        for re in conformed_events.drop_duplicates().itertuples())

    mode_occurrence = rl.groupby([
        'case_type', 'activity_type', 'time_type']).size().to_dict()

    F_conformed_res_events = 0.0
    F_allowed_res_events = 0.0
    F_model_false_res_events = 0.0
    for r in om.resources:
        for mode in om.find_execution_modes(r):
            res_event = (r, mode[0], mode[1], mode[2])
            if res_event in conformed_res_events:
                F_conformed_res_events += mode_occurrence[mode]
            else:
                F_model_false_res_events += mode_occurrence[mode]
            F_allowed_res_events += mode_occurrence[mode]

    return F_conformed_res_events / F_allowed_res_events


def precision(rl, om):
    """Calculate the precision of an organizational model against a 
    given event log. 
    
    Calculate model precision considering "allowed" events.

    Parameters
    ----------
    rl : DataFrame
        A resource log.
    om : OrganizationalModel
        The discovered organizational model.

    Returns
    -------
    float
        The result precision value.

    Notes
    -----
    A resource log instead of an event log is used here, however this 
    is valid since a resource log is (implicitly) one-to-one 
    corresponded with an event log.
    """
    cand_E = set()

    for event in rl.itertuples():
        if _is_allowed_event(event, om):
            m = (event.case_type, event.activity_type, event.time_type)
            cand_groups = om.find_candidate_groups(m)

            cand_e = frozenset.union(*cand_groups) # cand(e)
            cand_E.update(cand_e)
    n_cand_E = len(cand_E)

    if n_cand_E == 0:
        warn('No candidate resource.', RuntimeWarning)
        return float('nan')
    else:
        n_allowed_events = 0
        precision = 0.0

        for event in rl.itertuples():
            if _is_allowed_event(event, om):
                n_allowed_events += 1

                m = (event.case_type, event.activity_type, event.time_type)
                cand_groups = om.find_candidate_groups(m)
                cand_e = frozenset.union(*cand_groups) # cand(e)
                n_cand_e = len(cand_e)

                if _is_conformed_event(event, om):
                    # give reward
                    precision += (n_cand_E + 1 - n_cand_e) / n_cand_E
                else:
                    # NOTE: no extra penalty is given
                    precision += 0.0

        precision *= 1 / n_allowed_events
        return precision


def f1_score(rl, om):
    fitness_score = fitness(rl, om)
    precision_score = precision(rl, om)
    return ((2 * fitness_score * precision_score) 
            / (fitness_score + precision_score))

