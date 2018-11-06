#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')
import cProfile

fn_event_log = sys.argv[1]
fnout_org_model = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import NaiveActivityNameExecutionModeMiner
    naive_exec_mode_miner = NaiveActivityNameExecutionModeMiner(el)
    rl = naive_exec_mode_miner.derive_resource_log(el)

    # TODO: Timer related
    '''
    from time import time
    print('Timer active.')
    '''

    # discover organizational groups
    print('Input a number to choose a solution:')
    print('\t0. Default Mining (Song)')
    print('\t1. Metric based on Joint Activities/Cases (Song)')
    print('\t2. Hierarchical Organizational Mining')
    print('\t3. Overlapping Community Detection')
    print('\t4. Gaussian Mixture Model')
    print('\t5. Model based Overlapping Clustering')
    print('\t6. Fuzzy c-means')
    print('Option: ', end='')
    mining_option = int(input())

    if mining_option in []:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)

    elif mining_option == 0:
        from OrganizationalModelMiner.base import default_mining
        ogs = default_mining(rl)

    elif mining_option == 1:
        print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
        num_groups = input()
        num_groups = num_groups[1:-1].split(',')
        num_groups = range(int(num_groups[0]), int(num_groups[1]))

        # select method (MJA/MJC)
        print('Input a number to choose a method:')
        print('\t0. MJA')
        print('\t1. MJC')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            # build profiles
            from ResourceProfiler.raw_profiler import performer_activity_frequency
            profiles = performer_activity_frequency(rl, use_log_scale=False)
            from OrganizationalModelMiner.disjoint.graph_partitioning import (
                    mja)
            # MJA -> select metric (Euclidean distance/PCC)
            print('Input a number to choose a metric:')
            print('\t0. Distance (Euclidean)')
            print('\t1. PCC')
            print('Option: ', end='')
            metric_option = int(input())
            metrics = ['euclidean', 'correlation']
            ogs = mja(
                    profiles, num_groups, 
                    metric=metrics[metric_option])
        elif method_option == 1:
            from OrganizationalModelMiner.disjoint.graph_partitioning import (
                    mjc)
            ogs = mjc(el, num_groups)
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    elif mining_option == 2:
        print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
        num_groups = input()
        num_groups = num_groups[1:-1].split(',')
        num_groups = range(int(num_groups[0]), int(num_groups[1]))

        print('Input a number to choose a method:')
        print('\t0. Mining using cluster analysis')
        #print('\t1. Mining using community detection')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            # build profiles
            from ResourceProfiler.raw_profiler import performer_activity_frequency
            profiles = performer_activity_frequency(rl, use_log_scale=False)
            from OrganizationalModelMiner.hierarchical import clustering
            ogs, og_hcy = clustering.ahc(
                    profiles, num_groups, method='ward')
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)
        '''
        #TODO: [DEPRECATED]
        elif method_option == 1:
            # build social network
            #from SocialNetworkMiner.causality import handover
            from SocialNetworkMiner.joint_activities import distance
            sn = distance(el, use_log_scale=True, convert=True)
            from OrganizationalModelMiner.hierarchical import community_detection
            ogs, og_hcy = community_detection.betweenness(
                    sn, num_groups, weight='weight') # consider edge weight, optional
        '''
                
        og_hcy.to_csv(fnout_org_model + '_hierarchy')

    elif mining_option == 3:
        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=False)

        from OrganizationalModelMiner.overlap import community_detection
        print('Input a number to choose a method:')
        print('\t0. CFinder (Clique Percolation Method)') 
        print('\t1. LN + Louvain (Link partitioning)')
        print('\t2. OSLOM (Local expansion and optimization)')
        print('\t3. COPRA (Agent-based dynamical methods)')
        print('\t4. SLPA (Agent-based dynamical methods)')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            ogs = community_detection.clique_percolation(
                    profiles, metric='correlation')
        elif method_option == 1:
            ogs = community_detection.link_partitioning(
                    profiles, metric='correlation')
        elif method_option == 2:
            ogs = community_detection.local_expansion(
                    profiles, metric='correlation')
        elif method_option == 3:
            ogs = community_detection.agent_copra(
                    profiles, metric='correlation')
        elif method_option == 4:
            ogs = community_detection.agent_slpa(
                    profiles, metric='correlation')
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    elif mining_option == 4:
        print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
        num_groups = input()
        num_groups = num_groups[1:-1].split(',')
        num_groups = range(int(num_groups[0]), int(num_groups[1]))

        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=False)

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to use a random threshold):',
                end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        from OrganizationalModelMiner.overlap.clustering import gmm
        ogs = gmm(
                profiles, num_groups, threshold=user_selected_threshold)

        # TODO: Timer related
        '''
        tm_start = time()
        # execute
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)
        '''

    elif mining_option == 5:
        print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
        num_groups = input()
        num_groups = num_groups[1:-1].split(',')
        num_groups = range(int(num_groups[0]), int(num_groups[1]))

        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=False)

        from OrganizationalModelMiner.overlap.clustering import moc
        ogs = moc(
                profiles, num_groups)

        # TODO: Timer related
        '''
        tm_start = time()
        og = moc(profiles, num_groups, 
                warm_start_input_fn=ws_fn)
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)
        '''

    elif mining_option == 6:
        print('Input the desired range (e.g. [low, high)) of number of groups to be discovered:', end=' ')
        num_groups = input()
        num_groups = num_groups[1:-1].split(',')
        num_groups = range(int(num_groups[0]), int(num_groups[1]))

        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=False)

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to use a random threshold):',
                end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        from OrganizationalModelMiner.overlap.clustering import fcm
        ogs = fcm(
                profiles, num_groups, threshold=user_selected_threshold)

        # TODO: Timer related
        '''
        tm_start = time()
        og = fcm(profiles, num_groups,
                threshold=user_selected_threshold,
                warm_start_input_fn=ws_fn)
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)
        '''

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    from OrganizationalModelMiner.base import OrganizationalModel
    om = OrganizationalModel()

    # assign execution modes to groups
    from OrganizationalModelMiner.mode_assignment import assign_by_any
    from OrganizationalModelMiner.mode_assignment import assign_by_all
    for og in ogs:
        #modes = assign_by_any(og, rl)
        modes = assign_by_all(og, rl)

    from Evaluation.l2m import conformance
    print()
    print('Fitness\t\t= {:.6f}'.format(conformance.fitness(rl, om)))
    print('Precision\t= {:.6f}'.format(conformance.precision(rl, om)))

    # save the mined organizational model to a file
    with open(fnout_org_model, 'w', encoding='utf-8') as fout:
        om.to_file_csv(fout)
    print('\n[Org. model of {} resources in {} groups exported to "{}"]'
            .format(len(om.resources()), om.size(), fnout_org_model))

