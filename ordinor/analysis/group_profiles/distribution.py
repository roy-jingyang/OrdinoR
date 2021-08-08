"""
Resource group distribution

References
----------
.. [1] Yang, J., Ouyang, C., ter Hofstede, A., van der Aalst, W., &
Leyer, M. (2021). Seeing the Forest for the Trees: Group-Oriented
Workforce Analytics. In *Proceedings of the Business Process
Management 19th International Conference*, BPM 2021. Springer.
"""

import ordinor.constants as const

# NOTE: # (r, rg, q) / (*, rg, q)
def group_member_contribution(group, exe_ctx, rl):
    """
    Measure the member contribution of a group with respect to an
    execution context.

    Parameters
    ----------
    group : iterator
        Id of resources as a resource group.
    exe_ctx : 3-tuple
        An execution context.
    rl : pandas.DataFrame
        A resource log.

    Returns
    -------
    dict
        The measured member contribution for each of the group members.
    """
    # filtering irrelevant events
    rl = rl.loc[rl[const.RESOURCE].isin(group)].groupby([
        const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE]).get_group(exe_ctx)
    group_total_count = len(rl)

    group_load_distribution = dict()
    for r in group:
        group_load_distribution[r] = (
            len(rl.loc[rl[const.RESOURCE] == r]) / group_total_count)

    return group_load_distribution
