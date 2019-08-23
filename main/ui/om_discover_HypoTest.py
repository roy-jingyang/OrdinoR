#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout_org_model = sys.argv[2]

if __name__ == '__main__':
    print('This program is developed to test "Hypo" models.')

    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'(case) LoanGoal': 7})

    # discover organizational groups
    print('Input a number to choose a solution:')
    print('\t101. "Flower model" (BEST fitness, WORST precision)')
    print('\t102. "Enumerating model" (BEST fitness, BEST precision)')
    print('\t103. "Old Flower model" (BEST fitness, w/many exec. modes)')
    print('\t104. "rc-measure friendly model" (BEST rc-measure)')
    print('\t105. "Swap-1 model" (from enumerating model)')
    print('\t106. "Pick-1 model" (from enumerating model)')
    print('Option: ', end='')
    mining_option = int(input())

    if mining_option in []:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)

    elif mining_option == 101:
        # the "flower model"
        # Only 1 execution mode
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': '',
                'activity_type': '',
                'time_type': ''
            })

        from pandas import DataFrame
        rl = DataFrame(rl)

        # Only 1 resource group, containing ALL resources
        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))
        resources = set(rl['resource'].unique())
        ogs = [resources]

        # the only group is linked with the only execution mode
        from OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        for og in ogs:
            modes = frozenset([('', '', '')])
            om.add_group(og, modes)

    elif mining_option == 102:
        # the "enumerating model"
        # |E_res| execution modes (equal to #events with resource info)
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': '',
                'activity_type': event.Index,
                'time_type': ''
            })

        from pandas import DataFrame
        rl = DataFrame(rl)

        # |R| resource groups, i.e. each resource is a group by itself
        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))

        # each group is linked with modes (events) originated by the resource
        from OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        for r, events in rl.groupby('resource'):
            om.add_group(frozenset([r]), 
                frozenset((e.case_type, e.activity_type, e.time_type)
                    for e in events.itertuples()))

    elif mining_option == 103:
        #from ExecutionModeMiner.naive_miner import ATonlyMiner
        #naive_exec_mode_miner = ATonlyMiner(el)
        from ExecutionModeMiner.naive_miner import ATCTMiner
        naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')
        rl = naive_exec_mode_miner.derive_resource_log(el)

        # Only 1 resource group, containing ALL resources
        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))
        resources = set(rl['resource'].unique())
        ogs = [resources]

        from OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        # assign execution modes to groups
        from OrganizationalModelMiner.mode_assignment import assign_by_any
        for og in ogs:
            modes = assign_by_any(og, rl)
            om.add_group(og, modes)

    elif mining_option == 104:
        exit('Deprecated.')

        # NOTE 1: for a given log, more than 1 such models may be possible
        # NOTE 2: such models achieve best rc_measure (= 1.0), but not
        # necessarily best fitness (thus not comparable to the enumerating
        # model)
        # NOTE 3: the invention of such models needs to guarantee:
        #       (1) each execution mode has 1 capable resource and 1 only
        #       (2) all resources in the log are included

        # build profiles
        from ResourceProfiler.raw_profiler import count_execution_frequency
        profiles = count_execution_frequency(rl, scale='log')

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
            capable_resources = set(events['resource'])
            unused_cap_r = capable_resources.difference(set(ogs_d.keys()))
            if len(unused_cap_r) > 0:
                r = choice(list(unused_cap_r))
            else:
                r = choice(list(capable_resources))
            ogs_d[r].add(mode)

        ogs = list()
        modes_to_assign = list()
        for k, v in ogs_d.items():
            ogs.append(frozenset({k}))
            modes_to_assign.append(frozenset(v))

    elif mining_option == 105:
        # the "swap-1 model"
        # |E_res| execution modes (equal to #events with resource info)
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': '',
                'activity_type': event.Index,
                'time_type': ''
            })

        from pandas import DataFrame
        rl = DataFrame(rl)

        # |R| resource groups, i.e. each resource is a group by itself
        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))

        resources = sorted(set(rl['resource'].unique()))

        # pick a pair of resources as "candidates" for swapping
        # the first one and the last one
        res_p = resources[0]
        res_q = resources[-1]
        # each group is linked with modes (events) originated by the resource
        from OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        for r, events in rl.groupby('resource'):
            if r == res_p:
                om.add_group(frozenset([res_q]), 
                    frozenset((e.case_type, e.activity_type, e.time_type)
                        for e in events.itertuples()))
            elif r == res_q:
                om.add_group(frozenset([res_p]), 
                    frozenset((e.case_type, e.activity_type, e.time_type)
                        for e in events.itertuples()))
            else:
                om.add_group(frozenset([r]), 
                    frozenset((e.case_type, e.activity_type, e.time_type)
                        for e in events.itertuples()))

    elif mining_option == 106:
        # the "pick-1 model"
        # corresponding to the N_2 model in Sect. 6.4 (pp. 191)
        rl = list()
        for event in el.itertuples(): # keep order
            rl.append({
                'resource': event.resource,
                'case_type': '',
                'activity_type': event.Index,
                'time_type': ''
            })

        from pandas import DataFrame
        rl = DataFrame(rl)

        # 1 resource group, with only 1 resource in it
        from OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        e = rl.iloc[0]
        om.add_group(frozenset([e.resource]),
            frozenset({(e.case_type, e.activity_type, e.time_type)}))

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

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

    precision2_score = conformance.precision2(rl, om)
    print('Prec. (freq)\t= {:.6f}'.format(precision2_score))
    measure_values.append(precision2_score)
    print()

    precision1_score = conformance.precision1(rl, om)
    print('Prec. (no freq)\t= {:.6f}'.format(precision1_score))
    measure_values.append(precision1_score)
    print()

    '''
    precision3_score = conformance.precision3(rl, om)
    print('Prec. (new)\t= {:.6f}'.format(precision3_score))
    measure_values.append(precision3_score)
    print()
    '''

    precision4_score = conformance.precision4(rl, om)
    print('Prec. (new2)\t= {:.6f}'.format(precision4_score))
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

