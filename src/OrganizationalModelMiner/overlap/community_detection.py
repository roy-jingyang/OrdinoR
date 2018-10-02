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

# 1. Clique Percolation Method (CFinder by Pallas et al.) 
# TODO: why some of the nodes are missing in the result?
def clique_percolation(sn):
    '''
    This method implements the algorithm for discovering overlapping
    organizational models using community detection technique named clique
    percolation method.

    Notice that this is merely a "wrap-up" since the major procedure of the
    algorithm (i.e. community detection) is performed using the NetworkX module
    algorithms.community.kclique.k_clique_communities.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.

    Returns:
        list of sets
            A list of organizational groups.
    '''
    print('Applying overlapping organizational model mining using '
          'community detection (CFinder, Clique Percolation):')

    # step 1. Distinguish the isolated nodes
    from networkx import isolates
    original_isolates = list(isolates(sn))
    if len(original_isolates) > 0:
        print('[Warning] There exist {} ISOLATED NODES in the network.'.format(
            len(original_isolates)))

    # step 2. Run the detection algorithm
    from networkx.algorithms.community import k_clique_communities
    smallest_clique = 3
    groups = list(k_clique_communities(sn, smallest_clique))
    for iso in original_isolates:
        groups.append({iso})

    print('{} organizational groups discovered.'.format(len(groups)))
    return groups

# 2. Line Graph and Link Partitioning (Appice, based on Evans and Lambiotte)
def link_partitioning(sn):
    '''
    This method implements the three-phased algorithm for discovering over-
    lapping organizational models proposed by A.Appice, which involves trans-
    forming between original social networks and linear networks, and the app-
    lication of the Louvain community detection algorithm.

    Notice that due to scaling issue of linear network transforming, the
    Louvain algorithm application phase is processed using external SNA tools
    such as Pajek (preferred), Gephi, etc.

    The expected data exchange format is Pajek NET format.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.

    Returns:
        list of sets
            A list of organizational groups.
    '''

    print('Applying overlapping organizational model mining using '
          'community detection (Appice, Link partitioning):')

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

    with open('tmp_ln.net', 'w') as f_pajek_net:
        # header
        f_pajek_net.write('*Vertices {}\n'.format(len(edges)))
        # create nodes of the linear network (corresponding to edges)
        for i in range(len(edges)):
            e = edges[i]
            u = e[0]
            v = e[1]
            # "><" is used as deliminator to distinguish between nodes
            f_pajek_net.write('{} "{}><{}"\n'.format(i + 1, str(u), str(v)))

        print('{} nodes have been added to the linear network.'.format(
            len(edges)))

        # create edges of the linear network
        # header
        f_pajek_net.write('*arcs\n')
        cnt = 0
        for i in range(len(edges) - 1):
            ei = edges[i]
            for j in range(i + 1, len(edges)):
                ej = edges[j]

                x = None
                if ei[0] == ej[0] or ei[0] == ej[1]:
                    x = ei[0]
                elif ei[1] == ej[0] or ei[1] == ej[1]:
                    x = ei[1]

                if x is not None:
                    # i -> j
                    w_l = ei[2] / (
                            sn.degree(nbunch=x, weight='weight') - ej[2])
                    # i <- j
                    w_r = ej[2] / (
                            sn.degree(nbunch=x, weight='weight') - ei[2])

                    # precision of the edge weight value is 1e-9
                    f_pajek_net.write('{} {} {:.9f}\n'.format(
                        i + 1, j + 1, w_l))
                    f_pajek_net.write('{} {} {:.9f}\n'.format(
                        j + 1, i + 1, w_r))
                    cnt += 2

        print('{} edges have been added to the linear network.'.format(cnt))

    print('Transformed linear network exported to "tmp_ln.net".')
    print('Use an external SNA tool to discover communities:')

    # step 2. Run Louvain algorithm to discover communities
    # Run using external tool, then import the results back
    # Pajek (recommended): "Network" -> "Create Partitions" -> "Communities"
    # Gephi: "Statistics" -> "Modularity", result exported as *.gml
    print('Path to the community detection result file (*.clu) as input: ', 
            end='')
    fn_pajek_net_communities = input()
    print('Detected communities imported from "{}":'.format(
        fn_pajek_net_communities))

    # step 3. Map communities onto the original network to get the results
    # derive the orgnizational groups
    from collections import defaultdict
    groups = defaultdict(lambda: set())
    cnt = 0
    with open(fn_pajek_net_communities, 'r') as f:
        is_header_line = True
        for line in f:
            if is_header_line:
                is_header_line = False
            else:
                # one-to-one mapping between resource communities and linear
                # communities
                # TODO Optional: calculate the degree of membership
                label = line.strip()
                u = edges[cnt][0]
                v = edges[cnt][1]
                groups[label].add(u)
                groups[label].add(v)
                cnt += 1
        for i in range(len(original_isolates)):
            groups['ISOLATE #{}'.format(i)].add(original_isolates[i])

    print('{} organizational groups discovered.'.format(len(groups.values())))
    return [set(g) for g in groups.values()]

# 3. Local Expansion and Optimization (OSLOM by Lancichinetti et al.)
def local_expansion(sn):
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
    print('Detected communities imported from "{}":'.format(fn_communities))

    # step 4. Derive organizational groups from the detection results
    pass #TODO

# 4. Fuzzy Detection (MOSES by McDaid and Hurley)
def fuzzy_detection(sn):
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
    print('Use the external tool MOSES to discover communities:')

    # step 3. Run community detection applying MOSES
    print('Path to the community detection result file as input: ', end='')
    fn_communities = input()
    print('Detected communities imported from "{}":'.format(fn_communities))

    # step 4. Derive organizational groups from the detection results
    pass #TODO

# 5.1 Agent-based (COPRA by Gregory)
def agent_copra(sn):
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

    # step 3. Run community detection applying COPRA
    print('Path to the community detection result file as input: ', end='')
    fn_communities = input()
    print('Detected communities imported from "{}":'.format(fn_communities))

    # step 4. Derive organizational groups from the detection results
    pass #TODO

# 5.2 Agent-based (SLPA by Xie et al.)
def agent_slpa(sn):
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
    print('Use the external tool GANXiS (SLPA) to discover communities:')

    # step 3. Run community detection applying SLPA
    print('Path to the community detection result file as input: ', end='')
    fn_communities = input()
    print('Detected communities imported from "{}":'.format(fn_communities))

    # step 4. Derive organizational groups from the detection results
    pass #TODO

