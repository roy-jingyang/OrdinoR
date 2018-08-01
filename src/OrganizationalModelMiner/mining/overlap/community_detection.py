# -*- coding: utf-8 -*-

'''
This module contains the implementation of methods of mining overlapping orga-
nizational models, based on the use of community detection techniques. These
methods are SNA-based, meaning that a social network should be used as input.

Methods include:
    1. Louvain algorithm applied on linear network (A.Appice)
'''

def ln_louvain(sn):
    '''
    This method implements the three-phased algorithm for discoverying over-
    lapping organizational models proposed by A.Appice, which involves trans-
    forming between original social networks and linear networks, and the app-
    lication of the Louvain community detection algorithm.
    Notice that due to scaling issue of linear network transforming, the
    Louvain algorithm application phase is processed using external SNA tools
    such as Pajek (preferred), Gephi, etc.
    The expected data exchange format is Pajec NET format.

    Params:
        sn: NetworkX (Di)Graph
            A NetworkX (Di)Graph object, in which the resources are the nodes,
            and the edges could be connections built on similarties, inter-
            actions, etc.
    Returns:
        og: dict of sets
            The mined organizational groups.
    '''

    print('Applying overlapping organizational model mining using ' +
            'Appice\'s three-phased algorithm:')

    # step 1. Build the linear network using the original network
    edges = sorted(list(sn.edges.data('weight')))
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
        l_str_edges = list()
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
                    '''
                    f_pajek_net.write('{} {} {:.9f}\n'.format(
                        i + 1, j + 1, w_l))
                    f_pajek_net.write('{} {} {:.9f}\n'.format(
                        j + 1, i + 1, w_r))
                    '''
                    l_str_edges.append('{} {} {:.9f}\n'.format(
                        i + 1, j + 1, w_l))
                    l_str_edges.append('{} {} {:.9f}\n'.format(
                        j + 1, i + 1, w_r))
                    cnt += 2

        for i in range(cnt):
            f_pajek_net.write(l_str_edges[i])

        print('{} edges have been added to the linear network.'.format(cnt))

    print('Transformed linear network exported to "tmp_ln.net".')
    print('Use an external SNA tool to discover communities:')

    # step 2. Run Louvain algorithm to discover communities
    # Run using external tool, then import the results back in here
    # Pajek (recommended): "Network" -> "Create Partitions" -> "Communities"
    # Gephi: "Statistics" -> "Modularity", result exported as *.gml
    print('Path to the community detection result file (*.clu) as input: ', 
            end='')
    fn_pajek_net_communities = input()
    print('Detected communities imported from "{}":'.format(
        fn_pajek_net_communities))

    # 3. Map communities onto the original network to get the results
    # Derive the orgnizational model
    from collections import defaultdict
    og = defaultdict(lambda: set())
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
                og[label].add(u)
                og[label].add(v)
                cnt += 1

    print('{} organizational entities extracted.'.format(len(og)))
    from copy import deepcopy
    return deepcopy(og)

