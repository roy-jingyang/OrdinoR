"""
Resource group participation

References
----------
.. [1] Yang, J., Ouyang, C., ter Hofstede, A., van der Aalst, W., &
Leyer, M. (2021). Seeing the Forest for the Trees: Group-Oriented
Workforce Analytics. In *Proceedings of the Business Process
Management 19th International Conference*, BPM 2021. Springer.
"""

import ordinor.constants as const

# NOTE: R (r, rg, q) / (r, rg)
def group_coverage(group, exe_ctx, rl):
    """
    Measure the coverage of a group with respect to an execution context.

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
        The measured coverage.
    """
    # filtering irrelevant events
    rl = rl.loc[rl[const.RESOURCE].isin(group)]

    num_participants = 0
    for r in group:
        if len(rl.loc[
            (rl[const.RESOURCE] == r) &
            (rl[const.CASE_TYPE] == exe_ctx[0]) &
            (rl[const.ACTIVITY_TYPE] == exe_ctx[1]) &
            (rl[const.TIME_TYPE] == exe_ctx[2])]) > 0:
            num_participants += 1
        else:
            pass
    return num_participants / len(group)