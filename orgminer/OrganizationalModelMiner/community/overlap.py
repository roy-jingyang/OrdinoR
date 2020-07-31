# -*- coding: utf-8 -*-

"""This module contains the implementation of overlapping graph/network 
-based organizational mining methods, based on the use of community 
detection techniques [1]_.

References
----------
.. [1] Xie, J., Kelley, S., & Szymanski, B. K. (2013). Overlapping
   community detection in networks. *ACM Computing Surveys*, 45(4), 1–35.
   `<https://doi.org/10.1145/2501654.2501657>`_
"""
from deprecated import deprecated
from warnings import warn

def _relabel_nodes_integers(g):
    """A helper function that relabels nodes in a graph/network using 
    consecutive integer ids starting from 0. The relabeled ids are 
    ordered following the original alphabetical order as the original 
    node labels.

    Parameters
    ----------
    g : NetworkX Graph or DiGraph
        The original network of which the nodes are to be relabeled.

    Returns
    -------
    NetworkX Graph or DiGraph
        A new network with nodes relabeled.
    rev_mapping : dict
        A mapping from the new node ids (integers) to the original node 
        labels.

    See Also
    --------
    networkx.relabel.relabel_nodes
    """
    mapping = dict()
    rev_mapping = dict()
    for i, node in enumerate(sorted(g.nodes())):
        mapping[node] = i
        rev_mapping[i] = node
    from networkx import relabel_nodes
    return relabel_nodes(g, mapping), rev_mapping


def _extended_modularity(g, cover):
    """A helper function that calculates the extended modularity [1]_ 
    given a graph/network and a cover (i.e., a set of communities).

    Parameters
    ----------
    g : NetworkX Graph
        A graph/network.
    cover : list of sets
        A cover (a set of communities of nodes).
    
    Returns
    -------
    float
        The result extended modularity value.

    Notes
    -----
    The data attribute name of edge weight defaults to ``'weight'``, as
    in NetworkX.

    References
    ----------
    .. [1] Shen, H., Cheng, X., Cai, K., & Hu, M. B. (2009). Detect
       overlapping and hierarchical community structure in networks.
       *Physica A: Statistical Mechanics and its Applications*, 388(8),
       1706-1712. `<https://doi.org/10.1016/j.physa.2008.12.021>`_
    """
    from collections import defaultdict
    node_membership = defaultdict(set)
    # identify membership
    for i, community in enumerate(cover):
        for node in community:
            node_membership[node].add(i)

    # calculate extended modularity by iterating over every pair of
    # distint nodes within a community
    eq = 0.0
    m = sum([wt for (v, w, wt) in g.edges.data('weight')])
    from itertools import combinations
    for community in cover:
        for v, w in combinations(community, 2):
            Ov = len(node_membership[v])
            Ow = len(node_membership[w])
            Avw = g.edges[v, w]['weight'] if g.has_edge(v, w) else 0
            kv = g.degree(v, 'weight')
            kw = g.degree(w, 'weight')
            eq += (1.0 / (Ov * Ow) * (Avw - kv * kw / (2 * m)))
    return eq


def clique_percolation(profiles, metric='euclidean'):
    """Apply a clique percolation technique [1]_ for detecting 
    communities and thus to discover organizational groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    ogs : list of frozensets
        Discovered resource groups.
    
    Notes
    -----
    Using this function relies on an external tool, CFinder [1]_. This 
    function provides merely the interface between OrgMiner and the 
    external tool.
    
    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities
    scipy.spatial.distance

    References
    ----------
    .. [1] Palla, G., Derenyi, I., Farkas, I., & Vicsek, T. (2005).
       Uncovering the overlapping community structure of complex networks
       in nature and society. *Nature*, 435(7043), 814–818.
       `<https://doi.org/10.1038/nature03607>`_
    """
    print('Applying graph/network-based Clique Percolation Method ' + 
        '(overlapping community detection using CFinder):')
    # build network from profiles
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, rev_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        warn('There exist {} isolated nodes in the network.'.format(
            len(original_isolates)), RuntimeWarning)

    # step 2. Export network as edgelist
    from networkx import write_weighted_edgelist
    with open('tmp_sn.edgelist', 'wb') as f:
        write_weighted_edgelist(sn, f)
    print('Network exported to "tmp_sn.edgelist".')
    print('Use the external tool CFinder to discover communities:')

    # step 3. Run community detection applying CFinder
    print('Path to the output directory as input: ', end='')
    dirn_output = input()

    # step 4. Derive organizational groups from the detection results
    # Note: since CFinder produces a set of results varied by num. of cliques
    from collections import defaultdict
    best_fn = None
    best_eq = float('-inf')
    fn_cnt = 0
    solution = None
    from os import listdir, path
    for n in listdir(dirn_output):
        if path.isdir(path.join(dirn_output, n)) and n.startswith('k='):
            fn_cnt += 1
            fn_communities = path.join(dirn_output, n, 'communities')
            cnt = -1
            groups = defaultdict(set)
            with open(fn_communities, 'r') as f:
                for line in f:
                    if not (line == '' or line.startswith('#')):
                        cnt += 1
                        for label in line.split(':')[-1].strip().split():
                            groups[cnt].add(int(label))
            for i, iso_node in enumerate(original_isolates):
                groups['ISOLATE #{}'.format(i)].add(iso_node)
            eq = _extended_modularity(sn, list(groups.values()))
            if eq > best_eq:
                best_fn = fn_communities
                best_eq = eq
                solution = groups
    
    print('Detected communities imported from directory "{}":'.format(
        dirn_output))
    print('Best solution "{}" selected from {} candidates:'.format(
        best_fn, fn_cnt))
    print('{} organizational groups discovered.'.format(len(solution)))
    # restore labels
    ogs = list()
    for cover in solution.values():
        ogs.append(frozenset({rev_node_relabel_mapping[x] for x in cover}))
    return ogs


@deprecated(reason='This method requires a dependency unabled to be resolved.')
def link_partitioning(profiles, n_groups, metric='euclidean'):
    """Apply a link partitioning technique [1]_ for detecting communities 
    and thus to discover organizational groups [2]_.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    n_groups : int
        Expected number of resource groups.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    list of frozensets
        Discovered resource groups.

    Notes
    -----
    This function is an implementation of the method proposed in [2]_.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities
    scipy.spatial.distance

    References
    ----------
    .. [1] Evans, T. S., & Lambiotte, R. (2010). Line graphs of weighted
       networks for overlapping communities. *The European Physical
       Journal B*, 77(2), 265–272.
       `<https://doi.org/10.1140/epjb/e2010-00261-8>`_
    .. [2] Appice, A. (2017). Towards mining the organizational
       structure of a dynamic event scenario. *Journal of Intelligent
       Information Systems*, 1–29.
       `<https://doi.org/10.1007/s10844-017-0451-x>`_
    """
    print('Applying graph/network-based link partitioning method ' + 
        '(overlapping community detection):')
    raise NotImplementedError # TODO: dependency on louvain unresolved
    # build network from profiles
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)
    from orgminer.SocialNetworkMiner.utilities import select_edges_by_weight
    sn = select_edges_by_weight(sn, low=0) # Appice's setting
    sn = select_edges_by_weight(sn, percentage='+0.75') # Appice's setting

    # step 1. Build the linear network using the original network
    # distinguish the isolated nodes in the original network first
    # store as a Pajek .net format file as intermediate
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        warn('There exist {} isolated nodes in the network.'.format(
            len(original_isolates)), RuntimeWarning)
        print('When using an external tool to discover '
            'communities from the linear network, you should specify the'
            ' target number of communities as:\n\tN\' = N - {},'.format(
                len(original_isolates)), end=' ')
        print('where N is the actual target number to be obtained.')

    tmp_file_path = '.linear_graph.tmp'
    edges = sorted(list(sn.edges.data('weight')))
    with open(tmp_file_path, 'w') as f_pajek_net:
        # write header
        f_pajek_net.write('*Vertices {}\n'.format(len(edges)))

        # write nodes in LN
        for i, e in enumerate(edges):
            # "><" is used as deliminator to distinguish the original nodes
            f_pajek_net.write('{} "{}><{}"\n'.format(i + 1, 
                str(e[0]), str(e[1])))

        # write header
        f_pajek_net.write('*arcs\n')

        # write edges in LN
        from itertools import combinations
        for i, j in combinations(range(len(edges)), 2):
            ei = edges[i]
            ej = edges[j]
            joint = None
            if ei[0] == ej[0] or ei[0] == ej[1]:
                joint = ei[0]
            elif ei[1] == ej[0] or ei[1] == ej[1]:
                joint = ei[1]

            if joint is not None:
                # i -> j
                w_ij = (ei[2] 
                    / (sn.degree(nbunch=joint, weight='weight') - ej[2]))
                # i <- j
                w_ji = (ej[2] 
                    / (sn.degree(nbunch=joint, weight='weight') - ei[2]))

                # precision set to 1e-9
                f_pajek_net.write('{} {} {:.9f}\n'.format(i + 1, j + 1, w_ij))
                f_pajek_net.write('{} {} {:.9f}\n'.format(j + 1, i + 1, w_ji))

    # step 2. Run Louvain algorithm to discover communities
    # convert the graph to igraph format
    from igraph import Graph as iGraph
    ln_igraph = iGraph.Read_Pajek(tmp_file_path)
    from os import remove
    remove(tmp_file_path)

    import louvain
    louvain.set_rng_seed(0)
    optimiser = louvain.Optimiser()
    # search the resolution parameter using bisection
    lo = 0.5
    hi = 1.0
    eps = 0.01
    while (lo + hi) / 2 > eps:
        mid = (lo + hi) / 2
        partition = louvain.RBConfigurationVertexPartition(
            ln_igraph, weights='weight', resolution_parameter=mid)
        diff_inc = 1
        while diff_inc > 0:
            diff_inc = optimiser.optimise_partition(partition)
        ln_communities = partition

        if len(ln_communities) == n_groups:
            break
        elif len(ln_communities) < n_groups:
            lo = mid
        else:
            hi = mid

    # step 3. Map communities onto the original network to get the results
    # derive the orgnizational groups
    from collections import defaultdict
    groups = defaultdict(set)
    for i, comm in enumerate(ln_communities):
        for n_idx in comm:
            ln_v = ln_igraph.vs[n_idx]
            groups[i].add(ln_v['id'].split('><')[0])
            groups[i].add(ln_v['id'].split('><')[1])
    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(iso_node)
    return [frozenset(g) for g in groups.values()]


def local_expansion(profiles, metric='euclidean'):
    """Apply a local expansion technique [1]_ for detecting communities 
    and thus to discover organizational groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    list of frozensets
        Discovered resource groups.
    
    Notes
    -----
    Using this function relies on an external tool, OSLOM [1]_. This 
    function provides merely the interface between OrgMiner and the 
    external tool.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities
    scipy.spatial.distance

    References
    ----------
    .. [1] Lancichinetti, A., Radicchi, F., Ramasco, J. J., & Fortunato,
       S. (2011). Finding statistically significant communities in
       networks. *PloS one*, 6(4), e18961.
       `<https://doi.org/10.1371/journal.pone.0018961>`_
    """
    print('Applying graph/network-based local expansion method ' + 
        '(overlapping community detection using OSLOM):')
    # build network from profiles
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, rev_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        warn('There exist {} isolated nodes in the network.'.format(
            len(original_isolates)), RuntimeWarning)

    # step 2. Export network as edgelist
    from networkx import write_weighted_edgelist
    with open('tmp_sn.edgelist', 'wb') as f:
        write_weighted_edgelist(sn, f)
    print('Network exported to "tmp_sn.edgelist".')
    print('Use the external tool OSLOM to discover communities:')

    # step 3. Run community detection applying OSLOM
    print('Path to the community detection result file as input: ', end='')
    fn_communities = input()

    # step 4. Derive organizational groups from the detection results
    from collections import defaultdict
    groups = defaultdict(set)
    cnt = -1
    with open(fn_communities, 'r') as f:
        for line in f:
            if not line.startswith('#'): # for non-comment lines
                cnt += 1
                for label in line.split():
                    # restore labels 
                    groups[cnt].add(rev_node_relabel_mapping[int(label)])
    print('Detected communities imported from "{}":'.format(fn_communities))

    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(rev_node_relabel_mapping[iso_node])
    return [frozenset(g) for g in groups.values()]


def agent_copra(profiles, metric='euclidean'):
    """Apply an agent-based technique [1]_ for detecting communities 
    and thus to discover organizational groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    list of frozensets
        Discovered resource groups.
    
    Notes
    -----
    Using this function relies on an external tool, COPRA [1]_. This 
    function provides merely the interface between OrgMiner and the 
    external tool.
    
    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities
    scipy.spatial.distance
    agent_slpa

    References
    ----------
    .. [1] Gregory, S. (2010). Finding overlapping communities in
       networks by label propagation. *New Journal of Physics*, 12(10),
       103018. `<https://doi.org/10.1088/1367-2630/12/10/103018>`_
    """
    print('Applying graph/network-based agent-based method ' + 
        '(overlapping community detection using COPRA):')
    # build network from profiles
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, rev_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        warn('There exist {} isolated nodes in the network.'.format(
            len(original_isolates)), RuntimeWarning)

    # step 2. Export network as edgelist
    from networkx import write_weighted_edgelist
    with open('tmp_sn.edgelist', 'wb') as f:
        write_weighted_edgelist(sn, f)
    print('Network exported to "tmp_sn.edgelist".')
    print('Use the external tool COPRA to discover communities:')

    # step 3. Run community detection applying OSLOM
    print('Path to the community detection result file as input: ', end='')
    fn_communities = input()

    # step 4. Derive organizational groups from the detection results
    from collections import defaultdict
    groups = defaultdict(set)
    cnt = -1
    with open(fn_communities, 'r') as f:
        for line in f:
            cnt += 1
            for label in line.split():
                # restore labels 
                groups[cnt].add(rev_node_relabel_mapping[int(label)])
    print('Detected communities imported from "{}":'.format(fn_communities))

    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(rev_node_relabel_mapping[iso_node])
    return [frozenset(g) for g in groups.values()]


def agent_slpa(profiles, metric='euclidean'):
    """Apply an agent-based technique [1]_ for detecting communities 
    and thus to discover organizational groups.

    Parameters
    ----------
    profiles : DataFrame
        Constructed resource profiles.
    metric : str, optional, default 'euclidean'
        Choice of metrics for measuring the distance while calculating 
        distance. Defaults to ``'euclidean'``, meaning that euclidean
        distance is used for measuring distance.

    Returns
    -------
    ogs : list of frozensets
        Discovered resource groups.
    
    Notes
    -----
    Using this function relies on an external tool, GANXiSw [1]_. This 
    function provides merely the interface between OrgMiner and the 
    external tool.

    See Also
    --------
    orgminer.SocialNetworkMiner.joint_activities
    scipy.spatial.distance
    agent_copra

    References
    ----------
    .. [1] Xie, J., Szymanski, B. K., & Liu, X. (2011). Slpa: Uncovering
       overlapping communities in social networks via a speaker-listener
       interaction dynamic process. In *Proceedings of the 2011 IEEE 11th
       International Conference on data mining workshops*, pp. 344-349.
       IEEE. `<https://doi.org/10.1109/ICDMW.2011.154>`_
    """
    print('Applying graph/network-based agent-based method ' + 
        '(overlapping community detection using SLPA):')
    # build network from profiles
    from orgminer.SocialNetworkMiner.joint_activities import distance
    sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, rev_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        warn('There exist {} isolated nodes in the network.'.format(
            len(original_isolates)), RuntimeWarning)

    # step 2. Export network as edgelist
    from networkx import write_weighted_edgelist
    with open('tmp_sn.edgelist', 'wb') as f:
        write_weighted_edgelist(sn, f)
    print('Network exported to "tmp_sn.edgelist".')
    print('Use the external tool GANXiS (aka SLPA) to discover communities:')

    # step 3. Run community detection applying OSLOM
    print('Path to the output directory as input: ', end='')
    dirn_communities = input()

    # step 4. Derive organizational groups from the detection results
    # Note: since GANXiSw produces a set of results varied by the prob. "r"
    from collections import defaultdict
    best_fn = None
    best_eq = float('-inf')
    fn_cnt = 0
    solution = None
    from os import listdir, path
    for fn in listdir(dirn_communities):
        if fn.endswith('.icpm'):
            fn_cnt += 1
            fn_communities = path.join(dirn_communities, fn)
            cnt = -1
            groups = defaultdict(set)
            with open(fn_communities, 'r') as f:
                for line in f:
                    cnt += 1
                    for label in line.split():
                        groups[cnt].add(int(label))
            for i, iso_node in enumerate(original_isolates):
                groups['ISOLATE #{}'.format(i)].add(iso_node)
            eq = _extended_modularity(sn, list(groups.values()))
            if eq > best_eq:
                best_fn = fn_communities
                best_eq = eq
                solution = groups

    print('Detected communities imported from directory "{}":'.format(
        dirn_communities))
    print('Best solution "{}" selected from {} candidates:'.format(
        best_fn, fn_cnt))
    print('{} organizational groups discovered.'.format(len(solution)))
    # restore labels
    ogs = list()
    for cover in solution.values():
        ogs.append(frozenset({rev_node_relabel_mapping[x] for x in cover}))
    return ogs

