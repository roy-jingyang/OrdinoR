# -*- coding: utf-8 -*-

"""This module contains the implementation of the local diagnostics 
measures.
"""
# NOTE: # (*, rg, q) / (*, rg, *)
def group_relative_focus(group, mode, rl):
    """Measure the relative focus of a group with respect to an execution
    mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    mode : 3-tuple
        An execution mode.
    rl : DataFrame
        A resource log.

    Returns
    -------
    float
        The measured relative focus.
    """
    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)]
    grouped_by_modes = rl.groupby([
        'case_type', 'activity_type', 'time_type'])
    
    if mode in grouped_by_modes.groups:
        return len(grouped_by_modes.get_group(mode)) / len(rl)
    else:
        return 0.0


# NOTE: # (*, rg, q) / (*, *, q)
def group_relative_stake(group, mode, rl):
    """Measure the relative focus of a group with respect to an execution
    mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    mode : 3-tuple
        An execution mode.
    rl : DataFrame
        A resource log.

    Returns
    -------
    float
        The measured relative focus.
    """
    total_count = len(rl.groupby([
        'case_type', 'activity_type', 'time_type']).get_group(mode))

    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)]
    grouped_by_modes = rl.groupby([
        'case_type', 'activity_type', 'time_type'])
    if mode in grouped_by_modes.groups:
        return len(grouped_by_modes.get_group(mode)) / total_count
    else:
        return 0.0


# NOTE: R (r, rg, q) / (r, rg)
def group_coverage(group, mode, rl):
    """Measure the coverage of a group with respect to an execution mode.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    mode : 3-tuple
        An execution mode.
    rl : DataFrame
        A resource log.

    Returns
    -------
    float
        The measured coverage.
    """
    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)]

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


# NOTE: # (r, rg, q) / (*, rg, q)
def group_member_contribution(group, mode, rl):
    # filtering irrelevant events
    rl = rl.loc[rl['resource'].isin(group)].groupby([
        'case_type', 'activity_type', 'time_type']).get_group(mode)
    group_total_count = len(rl)

    group_load_distribution = dict()
    for r in group:
        group_load_distribution[r] = (
            len(rl.loc[rl['resource'] == r]) / group_total_count)

    return group_load_distribution

