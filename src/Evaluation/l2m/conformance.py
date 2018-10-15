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
    cand_groups = om.get_candidate_groups(m)
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
    '''
    count_conformed_events = 0 # "|E_conf|"
    for event in rl.itertuples():
        count_conformed_events += 1 if _is_conformed_event(event) else 0
    '''

    conformed_events = rl[rl.apply(
        lambda e: _is_conformed_event(e, om), axis=1)]
    n_conformed_events = len(conformed_events) # "|E_conf|"
    n_events = len(rl) # "|E_res|"
    return n_conformed_events / n_events

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
        cand_groups = om.get_candidate_groups(m)

        cand_e = frozenset.union(*cand_groups) # cand(e)
        l_n_cand_e.append(len(cand_e)) # "|cand(e)|"
        cand_E.update(cand_e) # update cand(E) by union with cand(e)

    n_cand_E = len(cand_E) # "|cand(E)|"

    if n_cand_E <= 1:
        print('[Warning] Number of overall set of candidate resources is '
              '{}.'.format(n_cand_E))
        return 1.0
    else:
        prec_sum = sum(
            [(n_cand_E - n_cand_e) / (n_cand_E - 1) for n_cand_e in l_n_cand_e])
        return prec_sum / n_conformed_events

# This is rather trivial ... (Should it even be implemented as a method?)
def f_score(rl, om, beta=1):
    v_fitness = fitness(rl, om)
    v_precision = precision(rl, om)
    score = (1 + beta ** 2) * (
            (v_precision * v_fitness)
            / ((beta ** 2) * v_precision + v_fitness))
    return score

