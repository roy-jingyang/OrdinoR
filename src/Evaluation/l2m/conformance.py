# -*- coding: utf-8 -*-

'''
This module contains the implmentation of the conformance checking measures
proposed in the OrgMining 2.0 framework.
'''
def _is_conformed_event(event, om):
    '''Determine whether an event in the resource log is conformining given an
    organizational model.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        event: row of DataFrame
            The event in the resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        boolean
            Boolean value indicating if the event is conformed with the model.
    '''
    m = (event.case_type, event.activity_type, event.time_type)
    cand_groups = om.find_candidate_groups(m)
    for g in cand_groups:
        if event.resource in g:
            return True

    return False

def fitness(rl, om):
    '''Calculate the fitness of an organizational model against a given
    resource log.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result fitness value.
    '''
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events) # "|E_conf|"
    n_events = len(rl) # "|E_res|"
    return n_conformed_events / n_events

def fitness1(rl, om):
    '''Calculate the fitness of an organizational model against a given
    resource log.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result fitness value.
    '''
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    # "|RE_conf|"
    n_conformed_res_events = len(conformed_events.drop_duplicates())
    # "|RE|" 
    n_actual_res_events = len(rl.drop_duplicates())
    return n_conformed_res_events / n_actual_res_events

def precision(rl, om):
    '''Calculate the precision of an organizational model against a given
    resource log, in which only the "fitting" (conformed) events are considered.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result precision value.
    '''
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
        cand_E.update(cand_e) # update cand(E) by union with cand(e)

    n_cand_E = len(cand_E) # "|cand(E)|"

    if n_cand_E == 0:
        print('[Warning] No candidate resource.')
        return float('nan')
    if n_cand_E == 1:
        print('[Warning] The overall number of candidate resources is 1.')
        return 1.0
    else:
        prec_sum = sum(
            [(n_cand_E - n_cand_e) / (n_cand_E - 1) for n_cand_e in l_n_cand_e])
        return prec_sum / n_conformed_events

def precision1(rl, om):
    '''Calculate the precision of an organizational model against a given
    resource log, in which only the "fitting" (conformed) events are considered.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result precision value.
    '''
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    # "|RE_conf|"
    n_conformed_res_events = len(conformed_events.drop_duplicates())
    # count of all possible (distinct) resource event allowed by the model
    n_allowed_res_events = sum(len(om.find_execution_modes(r))
            for r in om.resources())
    return n_conformed_res_events / n_allowed_res_events

# Chun's
def precision2(rl, om):
    '''Calculate the precision of an organizational model against a given
    resource log, in which only the "fitting" (conformed) events are considered.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result precision value.
    '''
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events)

    # RE_conf
    conformed_res_events = set(
            (re.resource, re.case_type, re.activity_type, re.time_type)
            for re in conformed_events.drop_duplicates().itertuples())

    # |RE(OM)|
    n_allowed_res_events = sum(len(om.find_execution_modes(r))
            for r in om.resources())

    def _mode_occurence(mode):
        return len(rl[
            (rl['case_type'] == mode[1]) &
            (rl['activity_type'] == mode[2]) &
            (rl['time_type'] == mode[3])])

    F_conformed_res_events = 0.0
    F_allowed_res_events = 0.0
    F_model_false_res_events = 0.0
    for r in om.resources():
        for mode in om.find_execution_modes(r):
            res_event = (r, mode[0], mode[1], mode[2])
            if res_event in conformed_res_events:
                pass
            else:
                F_model_false_res_events += _mode_occurence(res_event)
    F_model_false_res_events /= len(rl) * n_allowed_res_events

    return (1 - F_model_false_res_events)

# Roy's
def precision3(rl, om):
    '''Calculate the precision of an organizational model against a given
    resource log, in which only the "fitting" (conformed) events are considered.

    Note that a resource log instead of an event log is used here, however this
    should be valid since a resource log is (implicitly) one-to-one mapped from
    the corresponding event log.

    Params:
        rl: DataFrame
            The resource log.
        om: OrganizationalModel object
            The discovered organizational model.

    Returns:
        float
            The result precision value.
    '''
    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events)

    # RE_conf
    conformed_res_events = set(
            (re.resource, re.case_type, re.activity_type, re.time_type)
            for re in conformed_events.drop_duplicates().itertuples())

    def _mode_occurence(mode):
        return len(rl[
            (rl['case_type'] == mode[1]) &
            (rl['activity_type'] == mode[2]) &
            (rl['time_type'] == mode[3])])

    F_conformed_res_events = 0.0
    F_allowed_res_events = 0.0
    F_model_false_res_events = 0.0
    for r in om.resources():
        for mode in om.find_execution_modes(r):
            res_event = (r, mode[0], mode[1], mode[2])
            if res_event in conformed_res_events:
                F_conformed_res_events += _mode_occurence(res_event)
            else:
                F_model_false_res_events += _mode_occurence(res_event)
            F_allowed_res_events += _mode_occurence(res_event)

    return F_conformed_res_events / F_allowed_res_events

