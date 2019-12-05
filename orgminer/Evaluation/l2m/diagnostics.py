# -*- coding: utf-8 -*-

"""This module contains the implementation of the local diagnostics 
measures.
"""
# TODO: documentation
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

