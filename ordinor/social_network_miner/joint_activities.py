"""
Mining social networks from an event log using metrics based on joint
activities [1]_

See Also
--------
ordinor.social_network_miner.causality
ordinor.social_network_miner.joint_cases

References
----------
.. [1] Van der Aalst, W. M. P., Reijers, H. A., & Song, M. (2005).
   Discovering social networks from event logs. *Computer Supported
   Cooperative Work (CSCW)*, 14(6), 549-593.
   `<https://doi.org/10.1007/s10606-005-9005-9>`_
"""

from collections import defaultdict

from networkx import Graph, relabel_nodes
from numpy import log, fill_diagonal
from scipy.spatial.distance import squareform, pdist
from pandas import DataFrame

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

def performer_activity_matrix(el, use_log_scale):
    """
    Build a resource profile matrix solely based on how frequently
    resources originated activities, i.e., performer-by-activity matrix.

    Each column in the result profile matrix corresponds with an 
    activity captured in the given event log.

    Parameters
    ----------
    el : pandas.DataFrame, or pm4py EventLog
        An event log.
    use_log_scale : bool
        A Boolean flag indicating whether to apply logarithm scaling on 
        the values of the result profile matrix.

    Returns
    -------
    DataFrame
        The constructed resource profile matrix.

    See Also
    --------
    ordinor.org_model_miner.resource_features.direct_count
    """
    el = check_convert_input_log(el)
    pam = defaultdict(lambda: defaultdict(lambda: 0))
    for res, trace in el.groupby(const.RESOURCE):
        for event in trace.to_dict('records'):
            pam[res][event[const.ACTIVITY]] += 1

    if use_log_scale: 
        return DataFrame.from_dict(pam, orient='index').fillna(0).apply(
            lambda x: log(x + 1)
        )
    else:
        return DataFrame.from_dict(pam, orient='index').fillna(0)


def distance(profiles, metric='euclidean', convert=False):
    """
    Discover a social network from an event log based on joint activities
    where distance-related measures are used. 
    
    Parameters
    ----------
    profiles : pandas.DataFrame
        A resource profile matrix.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.
    convert : bool, optional, default False
        A Boolean flag indicating whether to convert the edge weight 
        values of the discovered network should be converted to 
        similarity measure values. Defaults to ``False``, i.e., keep as 
        distance measure.

    Returns
    -------
    sn : NetworkX Graph
        The discovered social network.

    Notes
    -----
    The edge weight values in a discovered social network correspond to 
    the distances between a pair of rows in the input profile matrix, 
    thus:

        1. A higher weight value indicates the two rows are less related. 
           This is different from rest of the social network discovery 
           metrics defined.
        2. The generated social network is undirected due to the nature 
           of distance (or similarity) measures.

    Refer to scipy.spatial.distance.pdist for more detailed explanation 
    of distance metrics.
    """
    x = squareform(pdist(profiles, metric=metric)) # preserve index

    if convert:
        exc.warn_runtime(
            'Distance measure converted to similarity measure',
        )

        # NOTE: different strategies may be employed for the conversion
        min_v = x.min()
        max_v = x.max()
        x = 1 - (x - min_v) / (max_v - min_v)

    fill_diagonal(x, 0) # ignore self-loops
    G = Graph(x)
    # relabel nodes using resource index in the profile matrix
    nodes = list(G.nodes)
    node_mapping = dict()
    for i in range(len(x)):
        node_mapping[nodes[i]] = profiles.index[i] 
    sn = relabel_nodes(G, node_mapping)
    sn.add_nodes_from(profiles.index)
    return sn
