#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

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
    print('\t3. Overlapping Community Detection (Appice)')
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
        # select method (MJA/MJC)
        print('Input a number to choose a method:')
        print('\t0. MJA')
        print('\t1. MJC')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            # MJA -> select metric (ED-distance/PCC)
            from SocialNetworkMiner import joint_activities
            print('Input a number to choose a metric:')
            print('\t0. Distance (Euclidean)')
            print('\t1. PCC')
            print('Option: ', end='')
            metric_option = int(input())
            if metric_option == 0:
                sn = joint_activities.distance(
                        el, use_log_scale=True, convert=True)
            elif metric_option == 1:
                sn = joint_activities.correlation(
                        el, use_log_scale=True)
            else:
                raise Exception('Failed to recognize input option!')
                exit(1)
        elif method_option == 1:
            from SocialNetworkMiner.joint_cases import working_together
            sn = working_together(el)
            print('[Warning] DiGraph casted to Graph.')
            sn = sn.to_undirected()
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

        # edge filtering
        print('Input a value as threshold:', end=' ')
        threshold = input()
        threshold = (threshold if threshold[0] in ['+', '-']
                else float(threshold))
        from SocialNetworkMiner.utilities import select_edges_by_weight
        if type(threshold) == float:
            sn = select_edges_by_weight(sn, low=threshold)
        else:
            sn = select_edges_by_weight(sn, percentage=threshold)

        # partitioning
        from OrganizationalModelMiner.disjoint import graph_partitioning
        ogs = graph_partitioning.connected_components(sn)

    elif mining_option == 2:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())

        print('Input a number to choose a method:')
        print('\t0. Mining using cluster analysis')
        print('\t1. Mining using community detection')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            # build profiles
            from ResourceProfiler.raw_profiler import performer_activity_frequency
            profiles = performer_activity_frequency(rl, use_log_scale=True)
            from OrganizationalModelMiner.hierarchical import cluster
            ogs, og_hcy = cluster.ahc(
                    profiles, num_groups, method='ward')
        elif method_option == 1:
            # build social network
            #from SocialNetworkMiner.causality import handover
            from SocialNetworkMiner.joint_activities import distance
            sn = distance(el, use_log_scale=True, convert=True)
            from OrganizationalModelMiner.hierarchical import community_detection
            ogs, og_hcy = community_detection.betweenness(
                    sn, num_groups, weight='weight') # consider edge weight, optional
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)
                
        og_hcy.to_csv(fnout_org_model + '_hierarchy')

    elif mining_option == 3:
        print('Input a number to choose a method:')
        print('\t0. Appice\'s algorithm (Linear Network + Louvain)')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            from OrganizationalModelMiner.overlap.community_detection import ln_louvain
            # build social network
            from SocialNetworkMiner.joint_activities import correlation
            sn = correlation(el, use_log_scale=True)

            # edge filtering
            print('Input a value as threshold:', end=' ')
            threshold = input()
            threshold = (threshold if threshold[0] in ['+', '-']
                    else float(threshold))
            from SocialNetworkMiner.utilities import select_edges_by_weight
            if type(threshold) == float:
                sn = select_edges_by_weight(sn, low=threshold)
            else:
                eps = sys.float_info.epsilon
                sn = select_edges_by_weight(sn, low=eps) # TODO: Appice only keeps the top positive values
                sn = select_edges_by_weight(sn, percentage=threshold)

            ogs = ln_louvain(sn)
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    elif mining_option == 4:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=True)
        #profiles = performer_activity_frequency(rl, use_log_scale=False)

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to choose the max., ' + 
                'creating disjoint groups):', end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        print('Input a number to choose a specific covariance type:')
        print('\t0. full 1. tied 2. diag 3. spherical (default)')
        cov_types = ['full', 'tied', 'diag', 'spherical']
        print('Option: ', end='')
        cov_type_option = input()
        cov_type_option = 3 if cov_type_option == '' else int(cov_type_option)

        print('Input a relative path to the file to be used for warm start: ' +
                '(Enter if None)')
        ws_fn = input()
        ws_fn = None if ws_fn == '' else ws_fn

        from OrganizationalModelMiner.overlap.cluster import gmm
        ogs = gmm(
                profiles, num_groups, threshold=user_selected_threshold,
                cov_type=cov_types[cov_type_option], warm_start_input_fn=ws_fn)

        # TODO: Timer related
        '''
        tm_start = time()
        # execute
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)
        '''

    elif mining_option == 5:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=True)
        #profiles = performer_activity_frequency(rl, use_log_scale=False)

        print('Input a relative path to the file to be used for warm start: ' +
                '(Enter if None)')
        ws_fn = input()
        ws_fn = None if ws_fn == '' else ws_fn

        from OrganizationalModelMiner.overlap.cluster import moc
        ogs = moc(
                profiles, num_groups, 
                warm_start_input_fn=ws_fn)

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
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from ResourceProfiler.raw_profiler import performer_activity_frequency
        profiles = performer_activity_frequency(rl, use_log_scale=True)
        #profiles = performer_activity_frequency(rl, use_log_scale=False)

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to choose the max., ' + 
                'creating disjoint groups):', end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        print('Input a relative path to the file to be used for warm start: ' +
                '(Enter if None)')
        ws_fn = input()
        ws_fn = None if ws_fn == '' else ws_fn

        from OrganizationalModelMiner.overlap.cluster import fcm
        ogs = fcm(
                profiles, num_groups,
                threshold=user_selected_threshold,
                warm_start_input_fn=ws_fn)

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
    from OrganizationalModelMiner.mode_assignment import group_first_assign
    for og in ogs:
        om.add_group(og, group_first_assign(og, rl))

    # TODO evaluate (goes here??)
    from Evaluation.l2m import conformance
    print()
    print('Fitness = {:.6f}'.format(conformance.fitness(rl, om)))
    print('Precision = {:.6f}'.format(conformance.precision(rl, om)))

    # save the mined organizational model to a file
    with open(fnout_org_model, 'w', encoding='utf-8') as fout:
        om.to_file_csv(fout)
    print('Organizational model exported to "{}".'.format(fnout_org_model))

