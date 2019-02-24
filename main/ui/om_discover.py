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
        #el = read_disco_csv(f, mapping={'(case) channel': 6})

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import ATonlyMiner
    from ExecutionModeMiner.naive_miner import ATCTMiner
    naive_exec_mode_miner = ATonlyMiner(el)
    #naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) channel')
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
    print('\t101. "One Group for All" (best fitness)')
    print('\t102. "One Group for Each" (best rc-measure)')
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
            from ResourceProfiler.raw_profiler import count_execution_frequency
            profiles = count_execution_frequency(rl, use_log_scale=False)
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
            profiles = count_execution_frequency(rl, use_log_scale=False)
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
        profiles = count_execution_frequency(rl, use_log_scale=False)

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
                    profiles, metric='euclidean')
        elif method_option == 1:
            ogs = overlap.link_partitioning(
                    profiles, metric='euclidean')
        elif method_option == 2:
            ogs = overlap.local_expansion(
                    profiles, metric='euclidean')
        elif method_option == 3:
            ogs = overlap.agent_copra(
                    profiles, metric='euclidean')
        elif method_option == 4:
            ogs = overlap.agent_slpa(
                    profiles, metric='euclidean')
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
        profiles = count_execution_frequency(rl, use_log_scale=False)

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
        profiles = count_execution_frequency(rl, use_log_scale=False)

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
        profiles = count_execution_frequency(rl, use_log_scale=False)

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
    elif mining_option == 101:
        # the "One Group for All" model (comparable to the flower model)

        # build profiles
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, use_log_scale=False)

        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))
        group = set(rl['resource'].unique())
        ogs = [group]
    elif mining_option == 102:
        # the "One Group for Each" model
        # NOTE 1: for a given log, more than 1 such models may be possible
        # NOTE 2: such models achieve best rc_measure (= 1.0), but not
        # necessarily best fitness (thus not comparable to the enumerating
        # model)
        # NOTE 3: the invention of such models needs to guarantee:
        #       (1) each execution mode has 1 capable resource and 1 only
        #       (2) all resources in the log are included

        # build profiles
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, use_log_scale=False)

        all_resources = set(rl['resource'].unique())
        from collections import defaultdict
        ogs_d = defaultdict(lambda: set())
        from numpy.random import choice
        grouped_by_mode = rl.groupby([
            'case_type', 'activity_type', 'time_type'])
        num_cand_of_modes = list((mode, len(set(events['resource'])))
                for mode, events in grouped_by_mode)
        # rarer mode first
        num_cand_of_modes.sort(key=lambda x: x[1])

        for mode, num_cand in num_cand_of_modes:
            # for each mode, find a capable resource
            events = grouped_by_mode.get_group(mode)
            capable_resources = list(set(events['resource']))
            while True:
                r = choice(capable_resources)
                if not r in ogs_d or len(capable_resources) == 1:
                    ogs_d[r].add(mode)
                    break
                else:
                    # try to cover more resources asap
                    capable_resources.remove(r)
        ogs = list()
        modes_to_assign = list()
        for k, v in ogs_d.items():
            ogs.append(frozenset({k}))
            modes_to_assign.append(frozenset(v))

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)


    from OrganizationalModelMiner.base import OrganizationalModel
    om = OrganizationalModel()

    if mining_option in [101, 102]:
        # handmade models
        if mining_option == 101:
            # the "One Group for All" model
            from OrganizationalModelMiner.mode_assignment import assign_by_any
            for og in ogs:
                modes = assign_by_any(og, rl)
                om.add_group(og, modes)
        elif mining_option == 102:
            # the "One Group for Each" model
            for i, og in enumerate(ogs):
                om.add_group(og, modes_to_assign[i])

    else:
        # assign execution modes to groups
        from OrganizationalModelMiner.mode_assignment import assign_by_any
        from OrganizationalModelMiner.mode_assignment import assign_by_all
        from OrganizationalModelMiner.mode_assignment import assign_by_proportion
        from OrganizationalModelMiner.mode_assignment import assign_by_weighting
        #jac_score = list()
        for og in ogs:
            modes = assign_by_any(og, rl)
            #modes = assign_by_all(og, rl)
            #modes = assign_by_proportion(og, rl, p=0.5)
            '''
            # TODO
            modes, jac = assign_by_weighting(og, rl, profiles)
            jac_score.append(jac)
            '''

            om.add_group(og, modes)

    print('-' * 80)
    measure_values = list()
    from Evaluation.m2m.cluster_validation import silhouette_score
    silhouette_score = silhouette_score(ogs, profiles)
    print('Silhouette\t= {:.6f}'.format(silhouette_score))
    print('-' * 80)
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
    precision3_score = conformance.precision3(rl, om)
    print('Prec. (freq)\t= {:.6f}'.format(precision3_score))
    measure_values.append(precision3_score)
    precision1_score = conformance.precision1(rl, om)
    print('Prec. (no freq)\t= {:.6f}'.format(precision1_score))
    measure_values.append(precision1_score)
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
    print()
    '''
    avg_jac_score = sum(jac_score) / len(jac_score)
    print('Avg. Jac.\t= {:.6f}'.format(avg_jac_score))
    '''
    print('-' * 80)
    print(','.join(str(x) for x in measure_values))


    # save the mined organizational model to a file
    with open(fnout_org_model, 'w', encoding='utf-8') as fout:
        om.to_file_csv(fout)
    print('\n[Org. model of {} resources in {} groups exported to "{}"]'
            .format(len(om.resources()), om.size(), fnout_org_model))

