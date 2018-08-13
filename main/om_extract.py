#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout_org_model = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    cases = read_disco_csv(fn_event_log)

    from time import time
    print('Timer active.')

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
        from OrganizationalModelMiner.mining import default_mining
        og = default_mining.mine(cases)

    elif mining_option == 1:
        from OrganizationalModelMiner.mining.disjoint import partition
        # select method (MJA/MJC)
        print('Input a number to choose a method:')
        print('\t0. MJA')
        print('\t1. MJC')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            # MJA -> select metric (ED-distance/PCC)
            from SocialNetworkMiner.mining import joint_activities
            print('Input a number to choose a metric:')
            print('\t0. Distance (Euclidean)')
            print('\t1. PCC')
            print('Option: ', end='')
            metric_option = int(input())
            if metric_option == 0:
                sn = joint_activities.distance(
                        cases, use_log_scale=True, convert=True)
            elif metric_option == 1:
                sn = joint_activities.correlation(
                        cases, use_log_scale=True)
            else:
                raise Exception('Failed to recognize input option!')
                exit(1)
        elif method_option == 1:
            from SocialNetworkMiner.mining.joint_cases import working_together
            sn = working_together(cases)
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
        from SocialNetworkMiner.mining.utilities import select_edges_by_weight
        if type(threshold) == float:
            sn = select_edges_by_weight(sn, low=threshold)
        else:
            sn = select_edges_by_weight(sn, percentage=threshold)

        # partitioning
        og = partition.connected_comp(sn)

    elif mining_option == 2:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())

        print('Input a number to choose a method:')
        print('\t0. Mining using cluster analysis')
        print('\t1. Mining using community detection')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            from OrganizationalModelMiner.mining.hierarchical import cluster
            # build profiles
            from SocialNetworkMiner.mining.joint_activities import build_performer_activity_matrix
            profiles = build_performer_activity_matrix(
                    cases, use_log_scale=True)
            og, og_hcy = cluster.ahc(profiles, num_groups,
                    method='ward')
        elif method_option == 1:
            from OrganizationalModelMiner.mining.hierarchical import community_detection
            # build social network
            #from SocialNetworkMiner.mining.causality import handover
            from SocialNetworkMiner.mining.joint_activities import distance
            sn = distance(cases, use_log_scale=True, convert=True)
            og, og_hcy = community_detection.betweenness(sn, num_groups,
                    weight='weight') # consider edge weight, optional
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
            from OrganizationalModelMiner.mining.overlap.community_detection import ln_louvain
            # build social network
            from SocialNetworkMiner.mining.joint_activities import correlation
            sn = correlation(cases)

            # edge filtering
            print('Input a value as threshold:', end=' ')
            threshold = input()
            threshold = (threshold if threshold[0] in ['+', '-']
                    else float(threshold))
            from SocialNetworkMiner.mining.utilities import select_edges_by_weight
            if type(threshold) == float:
                sn = select_edges_by_weight(sn, low=threshold)
            else:
                eps = sys.float_info.epsilon
                sn = select_edges_by_weight(sn, low=eps) # TODO: Appice only keeps the top positive values
                sn = select_edges_by_weight(sn, percentage=threshold)

            og = ln_louvain(sn)
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    elif mining_option == 4:
        from OrganizationalModelMiner.mining.overlap.cluster import gmm
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from SocialNetworkMiner.mining.joint_activities import build_performer_activity_matrix
        profiles = build_performer_activity_matrix(cases, use_log_scale=True)
        #profiles = build_performer_activity_matrix(cases, use_log_scale=False)

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

        tm_start = time()
        og = gmm(profiles, num_groups,
                threshold=user_selected_threshold,
                cov_type=cov_types[cov_type_option],
                warm_start_input_fn=ws_fn)
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)

    elif mining_option == 5:
        from OrganizationalModelMiner.mining.overlap.cluster import moc
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from SocialNetworkMiner.mining.joint_activities import build_performer_activity_matrix
        profiles = build_performer_activity_matrix(cases, use_log_scale=True)
        #profiles = build_performer_activity_matrix(cases, use_log_scale=False)

        print('Input a relative path to the file to be used for warm start: ' +
                '(Enter if None)')
        ws_fn = input()
        ws_fn = None if ws_fn == '' else ws_fn

        tm_start = time()
        og = moc(profiles, num_groups, 
                warm_start_input_fn=ws_fn)
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)

    elif mining_option == 6:
        from OrganizationalModelMiner.mining.overlap.cluster import fcm
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())
        # build profiles
        from SocialNetworkMiner.mining.joint_activities import build_performer_activity_matrix
        profiles = build_performer_activity_matrix(cases, use_log_scale=True)
        #profiles = build_performer_activity_matrix(cases, use_log_scale=False)

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

        tm_start = time()
        og = fcm(profiles, num_groups,
                threshold=user_selected_threshold,
                warm_start_input_fn=ws_fn)
        print('-' * 10
                + ' Execution time {:.3f} s. '.format(time() - tm_start)
                + '-' * 10)

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    from OrganizationalModelMiner import entity_assignment
    assignment = entity_assignment.assign(og, cases)

    # save the mined organizational model to a file
    from IO.writer import write_om_csv
    write_om_csv(fnout_org_model, og, assignment)
    #from IO.writer import write_om_omml

