# -*- coding: utf-8 -*-

'''
This module contains methods for associating discovered organizational groups
with execution modes.
'''

'''
The following (private) methods are used for assessing the relatedness between
a group and an execution mode regarding different factors (perspectives).
'''
def _participation_rate(group, mode, rl):
    '''Measure the participation rate of a group with respect to an execution
    mode.

    Params:
        group: iterator
            The ids of resources as a resource group.
        mode: tuple
            The execution mode.
        rl: DataFrame
            The resource log.
    Returns:
        : float
            The participated rate measured.
    '''
    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events

    total_par_count = len(rl)
    par_count = len(
        rl.loc[
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]
        )
            
    return par_count / total_par_count

def _coverage(group, mode, rl):
    '''Measure the coverage of a group with respect to an execution mode.

    Params:
        group: iterator
            The ids of resources as a resource group.
        mode: tuple
            The execution mode.
        rl: DataFrame
            The resource log.
    Returns:
        : float
            The coverage measured.
    '''
    rl = rl.loc[rl['resource'].isin(group)] # flitering irrelated events

    num_participants = 0
    for r in group:
        if len(rl.loc[
            (rl['resource'] == r) &
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]) > 0:
            num_participants += 1
        else:
            pass
    
    return num_participants / len(group)

'''
The following methods are for determining a set of execution modes for a given
group.
'''
def full_recall(group, rl):
    '''Assign an execution mode to a group, as long as there exists a member
    resource of this group that have executed this mode, i.e. everything done
    by each of the members matters.

    Note: this is the method proposed by Song & van der Aalst, DSS 2008,
    namely "entity_assignment".

    Params:
        group: iterator
            The ids of resources as a resource group.
        rl: DataFrame
            The resource log.
    Returns:
        modes: list
            The execution modes corresponded to the resources.
    '''
    print('Applying FullRecall for mode assignment:')
    modes = list()
    grouped_by_resource = rl.groupby('resource')

    for r in group:
        for event in grouped_by_resource.get_group(r).itertuples():
            m = (event.case_type, event.activity_type, event.time_type)
            if m not in modes:
                modes.append(m)

    return modes

def participation_first(group, rl, p):
    '''
    '''
    print('Applying ParticipationFirst with threshold {} '.format(p) +
        'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        par_rate = _participation_rate(group, m, rl)
        if par_rate >= p:
            tmp_modes.append((m, par_rate))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

def coverage_first(group, rl, p):
    '''
    '''
    print('Applying CoverageFirst with threshold {} '.format(p) +
        'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        coverage = _coverage(group, m, rl)
        if coverage >= p:
            tmp_modes.append((m, coverage))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

def overall_score(group, rl, p, w1=0.5, w2=0.5):
    '''
    '''
    print('Applying OverallScore with weights ({}, {}) '.format(w1, w2) +
        'and threshold {} '.format(p) + 'for mode assignment:')
    tmp_modes = list()
    all_execution_modes = set(rl[['case_type', 'activity_type', 'time_type']]
        .drop_duplicates().itertuples(index=False, name=None))

    for m in all_execution_modes:
        score = (w1 * _participation_rate(group, m, rl) + 
            w2 * _coverage(group, m, rl))
        if score >= p:
            tmp_modes.append((m, score))

    from operator import itemgetter
    modes = list(item[0] 
        for item in sorted(tmp_modes, key=itemgetter(1), reverse=True))
    return modes

