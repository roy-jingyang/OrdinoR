# -*- coding: utf-8 -*-

"""This module contains the implementation of the local diagnostics 
measures.
"""
# TODO: documentation
# NOTE: # (r, rg, *) / (*, rg, *)
# NOTE: this measure calculates the distribution of work happened by team
# members
def test_measure(rl, om):
    ogs = om.find_all_groups()
    from collections import defaultdict
    group_load = defaultdict(lambda: dict())

    grouped_by_resources = rl.groupby('resource')
    for og_id, og in ogs:
        for r in og:
            num_events = len(grouped_by_resources.get_group(r))
            group_load[og_id][r] = num_events

    for og_id, resource_counts in group_load.items():
        total_counts = sum(count for r, count in resource_counts.items())
        for r in resource_counts.keys():
            resource_counts[r] /= total_counts

    return group_load


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

    total_count = len(rl)
    count = len(
        rl.loc[
            (rl['case_type'] == mode[0]) &
            (rl['activity_type'] == mode[1]) &
            (rl['time_type'] == mode[2])]
    )
    return count / total_count


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

