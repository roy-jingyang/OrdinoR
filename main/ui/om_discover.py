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
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 8})

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.direct_groupby import ATonlyMiner
    from ExecutionModeMiner.direct_groupby import CTonlyMiner
    from ExecutionModeMiner.direct_groupby import ATCTMiner
    from ExecutionModeMiner.direct_groupby import ATTTMiner
    #naive_exec_mode_miner = ATonlyMiner(el)
    #naive_exec_mode_miner = CTonlyMiner(el, case_attr_name='product')
    #naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')
    naive_exec_mode_miner = ATTTMiner(el, resolution='day')

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
            from ResourceProfiler.raw_profiler import count_execution_frequency
            profiles = count_execution_frequency(rl, scale='log')
            from OrganizationalModelMiner.community.graph_partitioning import (
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
            from OrganizationalModelMiner.clustering.graph_partitioning import (
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
            from ResourceProfiler.raw_profiler import count_execution_frequency
            profiles = count_execution_frequency(rl, scale='log')
            from OrganizationalModelMiner.clustering.hierarchical import ahc
            ogs, og_hcy = ahc(
                    profiles, num_groups, method='ward')
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)
                
        og_hcy.to_csv(fnout_org_model + '_hierarchy')

    elif mining_option == 3:
        # build profiles
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, scale='log')

        from OrganizationalModelMiner.community import overlap
        print('Input a number to choose a method:')
        print('\t0. CFinder (Clique Percolation Method)') 
        print('\t1. LN + Louvain (Link partitioning)')
        print('\t2. OSLOM (Local expansion and optimization)')
        print('\t3. COPRA (Agent-based dynamical methods)')
        print('\t4. SLPA (Agent-based dynamical methods)')
        print('Option: ', end='')
        method_option = int(input())
        if method_option == 0:
            ogs = overlap.clique_percolation(
                    profiles, metric='correlation')
        elif method_option == 1:
            print('Input the desired number of groups to be discovered:', end=' ')
            num_groups = int(input())

            ogs = overlap.link_partitioning(
                    profiles, num_groups, metric='correlation')
        elif method_option == 2:
            ogs = overlap.local_expansion(
                    profiles, metric='correlation')
        elif method_option == 3:
            ogs = overlap.agent_copra(
                    profiles, metric='correlation')
        elif method_option == 4:
            ogs = overlap.agent_slpa(
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
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, scale='log')

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to use a "null" threshold):',
                end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        from OrganizationalModelMiner.clustering.overlap import gmm
        ogs = gmm(
                profiles, num_groups, threshold=user_selected_threshold,
                init='kmeans')

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
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, scale='log')

        from OrganizationalModelMiner.clustering.overlap import moc
        ogs = moc(
                profiles, num_groups,
                init='kmeans')

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
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, scale='log')

        print('Input a threshold value [0, 1), in order to determine the ' +
                'resource membership (Enter to use a "null" threshold):',
                end=' ')
        user_selected_threshold = input()
        user_selected_threshold = (float(user_selected_threshold)
                if user_selected_threshold != '' else None)

        from OrganizationalModelMiner.clustering.overlap import fcm
        ogs = fcm(
                profiles, num_groups, threshold=user_selected_threshold,
                init='kmeans')

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
    from OrganizationalModelMiner.mode_assignment import assign_by_proportion
    from OrganizationalModelMiner.mode_assignment import assign_by_weighting
    for og in ogs:
        modes = assign_by_any(og, rl)
        #modes = assign_by_all(og, rl)
        #modes = assign_by_proportion(og, rl, p=0.5)
        #modes = assign_by_weighting(og, rl, profiles)

        om.add_group(og, modes)

    print('-' * 80)
    measure_values = list()
    '''
    from Evaluation.m2m.cluster_validation import silhouette_score
    silhouette_score = silhouette_score(ogs, profiles)
    print('Silhouette\t= {:.6f}'.format(silhouette_score))
    print('-' * 80)
    '''
    print()
    
    from Evaluation.l2m import conformance
    fitness_score = conformance.fitness(rl, om)
    print('Fitness\t\t= {:.6f}'.format(fitness_score))
    measure_values.append(fitness_score)
    print()

    rc_measure_score = conformance.rc_measure(rl, om)
    print('rc-measure\t= {:.6f}'.format(rc_measure_score))
    measure_values.append(rc_measure_score)
    print()

    #print('Fitness1\t= {:.6f}'.format(conformance.fitness1(rl, om)))
    precision2_score = conformance.precision2(rl, om)
    print('Prec. (freq)\t= {:.6f}'.format(precision2_score))
    measure_values.append(precision2_score)

    precision1_score = conformance.precision1(rl, om)
    print('Prec. (no freq)\t= {:.6f}'.format(precision1_score))
    measure_values.append(precision1_score)

    precision4_score = conformance.precision4(rl, om)
    print('Prec. (new)\t= {:.6f}'.format(precision4_score))
    measure_values.append(precision4_score)
    print()

    # Overlapping Density & Overlapping Diversity (avg.)
    k = om.size()
    resources = om.resources()
    n_ov_res = 0
    n_ov_res_membership = 0
    for r in resources:
        n_res_membership = len(om.find_groups(r))
        if n_res_membership == 1:
            pass
        else:
            n_ov_res += 1
            n_ov_res_membership += n_res_membership

    ov_density = n_ov_res / len(resources)
    avg_ov_diversity = (n_ov_res_membership / n_ov_res 
            if n_ov_res > 0 else float('nan'))
    print('Ov. density\t= {:.6f}'.format(ov_density))
    print('Ov. diversity\t= {:.6f}'.format(avg_ov_diversity))
    measure_values.append(ov_density)
    measure_values.append(avg_ov_diversity)
    print('-' * 80)
    print(','.join(str(x) for x in measure_values))


    # save the mined organizational model to a file
    with open(fnout_org_model, 'w', encoding='utf-8') as fout:
        om.to_file_csv(fout)
    print('\n[Org. model of {} resources in {} groups exported to "{}"]'
            .format(len(om.resources()), om.size(), fnout_org_model))

