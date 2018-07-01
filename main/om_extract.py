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

    print('Input a number to choose a solution:')
    print('\t0. Default Mining (Song)')
    print('\t1. Metric based on Joint Activities (Song)')
    print('\t2. Hierarchical Organizational Mining')
    print('\t3. Overlapping Community Detection (Appice)')
    print('\t4. Gaussian Mixture Model')
    print('\t5. Model based Overlapping Clustering')
    print('Option: ', end='')
    mining_option = int(input())


    if mining_option in [3, 4, 5]:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)
    elif mining_option == 0:
        from OrganizationalModelMiner.mining import default_mining
        og = default_mining.mine(cases)

    elif mining_option == 2:
        print('Input a integer for the desired number of groups to be discovered:', end=' ')
        num_groups = int(input())

        print('Input a number to choose a method:')
        print('\t0. Mining using cluster analysis')
        print('\t1. Mining using community detection')
        method_option = int(input())
        if method_option == 0:
            from OrganizationalModelMiner.mining.hierarchical import cluster
            # build profiles
            from SocialNetworkMiner.mining.joint_activities import \
                    build_performer_activity_matrix
            profiles = build_performer_activity_matrix(
                    cases, use_log_scale=False)
            og, og_hcy = cluster.ahc(profiles, num_groups)
        elif method_option == 1:
            from OrganizationalModelMiner.mining.hierarchical import \
                    community_detection
            # build social network
            #from SocialNetworkMiner.mining.causality import handover
            from SocialNetworkMiner.mining.joint_activities import distance
            sn = distance(cases)
            og, og_hcy = community_detection.betweenness(sn, num_groups,
                    weight='weight') # consider edge weight
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)
                
        og_hcy.to_csv(fnout_org_model + '_hierarchy')

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    '''
    if mining_option.split('.')[0] == 'task':
        elif mining_option.split('.')[1] == 'mja':
            exit(1)
            #threshold_value_step = float(additional_params[0])
            from mining.HardClustering import MJA
            og = MJA.threshold(cases)
            #og = MJA.threshold(cases, threshold_value_step)
        elif mining_option.split('.')[1] == 'community':
            exit(1)
            from mining.SoftClustering import Community
            f_sn_model = additional_params[0]
            og = Community.mine(f_sn_model)
    '''
    '''
    elif mining_option.split('.')[0] == 'case':
        raise Exception('Case-based mining under construction!')
        exit(1)
        from mining.HardClustering import MJC
        if mining_option.split('.')[1] == 'mjc_threshold':
            threshold_value = float(additional_params[0])
            og = MJC.threshold(cases, threshold_value)
        elif mining_option.split('.')[1] == 'mjc_remove':
            min_centrality = float(additional_params[0])
            if mining_option.split('.')[2] == 'degree':
                og = MJC.remove_by_degree(cases, min_centrality)
            elif mining_option.split('.')[2] == 'betweenness':
                og = MJC.remove_by_betweenness(cases, min_centrality)
            else:
                raise Exception('Option for case-based mining invalid.')
        else:
            raise Exception('Option for case-based mining invalid.')
    else:
        raise Exception('Failed to recognize input parameter!')
        exit(1)
    '''

    from OrganizationalModelMiner import entity_assignment
    assignment = entity_assignment.assign(og, cases)

    # save the mined organizational model to a file
    from IO.writer import write_om_csv
    write_om_csv(fnout_org_model, og, assignment)
    #from IO.writer import write_om_omml

    # if hierarchical organizational mining, save the hierarchy as well
    

