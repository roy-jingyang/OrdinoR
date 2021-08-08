"""
Resource group workload

References
----------
.. [1] Yang, J., Ouyang, C., ter Hofstede, A., van der Aalst, W., &
Leyer, M. (2021). Seeing the Forest for the Trees: Group-Oriented
Workforce Analytics. In *Proceedings of the Business Process
Management 19th International Conference*, BPM 2021. Springer.
"""

import ordinor.constants as const

# NOTE: # (*, rg, q) / (*, rg, *)
def group_relative_focus(group, exe_ctx, rl):
    """
    Measure the relative focus of a group with respect to an execution
    context.

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
    float
        The measured relative focus.
    """
    # filtering irrelevant events
    rl = rl.loc[rl[const.RESOURCE].isin(group)]
    grouped_by_ctx = rl.groupby([
        const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE])
    
    if exe_ctx in grouped_by_ctx.groups:
        return len(grouped_by_ctx.get_group(exe_ctx)) / len(rl)
    else:
        return 0.0


# NOTE: # (*, rg, q) / (*, *, q)
def group_relative_stake(group, exe_ctx, rl):
    """
    Measure the relative focus of a group with respect to an execution
    context.

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
    float
        The measured relative focus.
    """
    total_count = len(rl.groupby([
        const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE]).get_group(exe_ctx))

    # filtering irrelevant events
    rl = rl.loc[rl[const.RESOURCE].isin(group)]
    grouped_by_ctx = rl.groupby([
        const.CASE_TYPE, const.ACTIVITY_TYPE, const.TIME_TYPE])
    if exe_ctx in grouped_by_ctx.groups:
        return len(grouped_by_ctx.get_group(exe_ctx)) / total_count
    else:
        return 0.0
