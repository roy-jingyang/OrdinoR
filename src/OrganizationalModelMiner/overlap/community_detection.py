# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining overlapping orga-
nizational models, based on the use of community detection techniques. These
methods are SNA-based, meaning that a social network should be used as input.

Methods include:
    1. Clique Percolation Method (CFinder by Pallas et al.) 
    2. Line Graph and Link Partitioning (Appice, based on Evans and Lambiotte)
    3. Local Expansion and Optimization (OSLOM by Lancichinetti et al.)
    4. Fuzzy Detection (MOSES by McDaid and Hurley)
    5. Agent-based and Dynamical Algorithms
        5.1 COPRA (by Gregory)
        5.2 SLPA(w) (aka GANXiS by Xie et al.)
'''

def _relabel_nodes_integers(g):
    '''
    This method is a utility function that relabels the nodes in a network as
    consequtive integers starting from 0, following the alphabetical order of
    the original node labels.

    Params:
        g: NetworkX (Di)Graph
            A Network (Di)Graph object, the original network. 

    Returns:
        NetworkX (Di)Graph
            A Network (Di)Graph object, the relabeled network. 
        
        inv_mapping: dict
            The inverse label mapping used for later recovering the label of
            nodes.
    '''
    mapping = dict()
    inv_mapping = dict()
    for i, node in enumerate(sorted(g.nodes())):
        mapping[node] = i
        inv_mapping[i] = node
    from networkx import relabel_nodes
    return relabel_nodes(g, mapping), inv_mapping

def _extended_modularity(g, cover):
    '''
    This method is a utility function that calculates the extended modularity
    (proposed by Shen et al. (2009) "Detect overlapping and hierarchical
    community structure in networks", Physica, with a network and the
    discovered cover (i.e. communities) given.

    Params:
        g: NetworkX (Di)Graph
            A Network (Di)Graph object, the original network. 
        cover: list of sets
            The discovered communities of nodes.

    Returns:
        float
            The calculated extended modularity by definition.
    '''
    from collections import defaultdict
    node_membership = defaultdict(lambda: set())
    # identify membership
    for i, community in enumerate(cover):
        for node in community:
            node_membership[node].add(i)

    # calculate extended modularity by iterating over every pair of distint
    # nodes within a community
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

# 1. Clique Percolation Method (CFinder by Pallas et al.) 
def clique_percolation(
        profiles,
        metric='euclidean', use_log_scale=False):
    '''
    This method implements the algorithm for discovering overlapping
    organizational models using a community detection technique named clique
    percolation method.
    
    The implementation is done using the external software CFinder. The number
    of communities to be discovered is determined automatically.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using '
          'community detection (CFinder from Clique Percolation methods):')
    # build network from profiles
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, inv_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the network.'.format(
            len(original_isolates)))

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
            groups = defaultdict(lambda: set())
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
    og = list()
    for cover in solution.values():
        og.append(frozenset({inv_node_relabel_mapping[x] for x in cover}))

    return og

# 2. Line Graph and Link Partitioning (Appice, based on Evans and Lambiotte)
def link_partitioning(
        profiles,
        metric='euclidean', use_log_scale=False):
    '''
    This method implements the three-phased algorithm for discovering over-
    lapping organizational models proposed by A.Appice, which involves trans-
    forming from an original social networks to a linear network, and the app-
    lication of the Louvain community detection algorithm.

    The implementation is done using NetworkX built-in methods and the
    python-louvain module.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''

    print('Applying overlapping organizational model mining using '
          'community detection (Appice\'s method from Link partitioning):')
    # build network from profiles
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)

    # step 1. Build the linear network using the original network
    edges = sorted(list(sn.edges.data('weight')))

    # distinguish the isolated nodes in the original network first
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the original network.'
                .format(len(original_isolates)), end=' ')
        print('This indicates that when using any external tools to discover '
              'communities from the linear network, you should specify the '
              'target number of communities as:\n\tN\' = N - {},'.format(
                  len(original_isolates)), end=' ')
        print('where N is the actual target number to be obtained in the '
              'final result.')

    from networkx import DiGraph
    ln = DiGraph()
    from itertools import combinations
    for ei, ej in list(combinations(sn.edges.data('weight'), 2)):
        joint = None
        if ei[0] == ej[0] or ei[0] == ej[1]:
            joint = ei[0]
        elif ei[1] == ej[0] or ei[1] == ej[1]:
            joint = ei[1]

        if joint is not None:
            # i -> j
            w_ij = ei[2] / (sn.degree(nbunch=joint, weight='weight') - ej[2])
            # i <- j
            w_ji = ej[2] / (sn.degree(nbunch=joint, weight='weight') - ei[2])

            ln.add_edge(
                    '{}***{}'.format(ei[0], ei[1]),
                    '{}***{}'.format(ej[0], ej[1]),
                    weight=w_ij)
            ln.add_edge(
                    '{}***{}'.format(ej[0], ej[1]),
                    '{}***{}'.format(ei[0], ei[1]),
                    weight=w_ji)

    # step 2. Run Louvain algorithm to discover communities
    from community import best_partition
    # TODO: casting to undirected network
    ln = ln.to_undirected()
    ln_communities = best_partition(ln)

    # step 3. Map communities onto the original network to get the results
    # derive the orgnizational groups
    from collections import defaultdict
    groups = defaultdict(lambda: set())
    # one-to-one mapping between resource communities and linear
    # communities
    # TODO Optional: calculate the degree of membership
    for e, comm in ln_communities.items():
            groups[comm].add(e.split('***')[0])
            groups[comm].add(e.split('***')[1])
    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(iso_node)

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [frozenset(g) for g in groups.values()]

# 3. Local Expansion and Optimization (OSLOM by Lancichinetti et al.)
def local_expansion(
        profiles,
        metric='euclidean', use_log_scale=False):
    '''
    This method implements the algorithm for discovering overlapping
    organizational models using community detection technique named OSLOM,
    which is a local expansion and optimization method.

    This implementation is done using the external software OSLOM. The number
    of communities to be discovered is determined automatically.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using '
          'community detection (OSLOM from Local expansion methods):')
    # build network from profiles
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, inv_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the network.'.format(
            len(original_isolates)))

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
    groups = defaultdict(lambda: set())
    cnt = -1
    with open(fn_communities, 'r') as f:
        for line in f:
            if not line.startswith('#'): # for non-comment lines
                cnt += 1
                for label in line.split():
                    # restore labels 
                    groups[cnt].add(inv_node_relabel_mapping[int(label)])
    print('Detected communities imported from "{}":'.format(fn_communities))

    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(inv_node_relabel_mapping[iso_node])

    return [frozenset(g) for g in groups.values()]

# 5.1 Agent-based (COPRA by Gregory)
def agent_copra(
        profiles,
        metric='euclidean', use_log_scale=False):
    '''
    This method implements the algorithm for discovering overlapping
    organizational models using community detection technique named COPRA,
    which is a agent-based dynamical method.

    This implementation is done using the external software COPRA. The number
    of communities to be discovered is determined automatically.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using '
          'community detection (COPRA from Agent-based methods):')
    # build network from profiles
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, inv_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the network.'.format(
            len(original_isolates)))

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
    groups = defaultdict(lambda: set())
    cnt = -1
    with open(fn_communities, 'r') as f:
        for line in f:
            cnt += 1
            for label in line.split():
                # restore labels 
                groups[cnt].add(inv_node_relabel_mapping[int(label)])
    print('Detected communities imported from "{}":'.format(fn_communities))

    for i, iso_node in enumerate(original_isolates):
        groups['ISOLATE #{}'.format(i)].add(inv_node_relabel_mapping[iso_node])

    return [frozenset(g) for g in groups.values()]

# 5.2 Agent-based (SLPA by Xie et al.)
# TODO
def agent_slpa(
        profiles,
        metric='euclidean', use_log_scale=False):
    '''
    This method implements the algorithm for discovering overlapping
    organizational models using community detection technique named SLPA,
    which is a agent-based dynamical method.

    This implementation is done using the external software GANXiSw. The number
    of communities to be discovered is determined automatically.

    Params:
        profiles: DataFrame
            With resource ids as indices and activity names as columns, this
            DataFrame contains profiles of the specific resources.
        metric: str, optional
            Choice of metrics for measuring the distance while calculating the
            linkage. Refer to scipy.spatial.distance.pdist for more detailed
            explanation.
        use_log_scale: boolean
            Use the logrithm scale if the volume of work varies significantly.
    Returns:
        list of frozensets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using '
          'community detection (SLPA from Agent-based methods):')
    # build network from profiles
    correlation_based_metrics = ['pearson']
    if metric in correlation_based_metrics:
        from SocialNetworkMiner.joint_activities import correlation
        sn = correlation(profiles, metric=metric, convert=True) 
    else:
        from SocialNetworkMiner.joint_activities import distance
        sn = distance(profiles, metric=metric, convert=True)

    # step 0. Relabel nodes
    sn, inv_node_relabel_mapping = _relabel_nodes_integers(sn)

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the network.'.format(
            len(original_isolates)))

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
            groups = defaultdict(lambda: set())
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
    og = list()
    for cover in solution.values():
        og.append(frozenset({inv_node_relabel_mapping[x] for x in cover}))

    return og

