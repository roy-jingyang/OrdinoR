#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fnout_org_model = sys.argv[2]

if __name__ == '__main__':
    print('This program is developed to test "Hypo" models.')

    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # discover organizational groups
    print('Input a number to choose a solution:')
    print('\t101. "Flower model" (BEST fitness, WORST precision)')
    print('\t102. "Enumerating model" (BEST fitness, BEST precision)')
    print('\t103. "Old Flower model" (BEST fitness, w/many exec. modes)')
    print('\t104. "rc-measure friendly model" (BEST rc-measure)')
    print('\t105. "Swap-1 model" (from enumerating model)')
    print('\t106. "Pick-1 model" (from enumerating model)')
    print('\t107. "Enumerating model (Res. Events) (BEST fitness, ' +
        'BEST possible precision under a fixed set of exec. modes)')
    print('\t108. "Pick-many model (from enumerating model, RE) ' +
            '(Sacrificing the least fitness to get BEST possible' +
            'precision under a fixed set of exec. modes)')
    print('Option: ', end='')
    mining_option = int(input())

    if mining_option in []:
        raise RuntimeError('Options unavailable.')

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
        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
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
        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        for r, events in rl.groupby('resource'):
            om.add_group({r}, list(events[
                ['case_type', 'activity_type', 'time_type']].itertuples(
                    index=False)))

    elif mining_option == 103:
        #from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
        #mode_miner = ATonlyMiner(el)
        from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
        mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')
        rl = mode_miner.derive_resource_log(el)

        # Only 1 resource group, containing ALL resources
        print('Total Num. resource event in the log: {}'.format(
            len(rl.drop_duplicates())))
        resources = set(rl['resource'].unique())
        ogs = [resources]

        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        # assign execution modes to groups
        from orgminer.OrganizationalModelMiner.mode_assignment import full_recall
        for og in ogs:
            modes = full_recall(og, rl)
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
        from orgminer.ResourceProfiler.raw_profiler import count_execution_frequency
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
        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
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
        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        e = rl.iloc[0]
        om.add_group(frozenset([e.resource]),
            frozenset({(e.case_type, e.activity_type, e.time_type)}))

    elif mining_option == 107:
        # the "enumerating model" but w.r.t. a fixed set of exec. modes
        from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
        from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
        mode_miner = ATonlyMiner(el)

        rl = mode_miner.derive_resource_log(el)

        from orgminer.OrganizationalModelMiner.base import OrganizationalModel
        om = OrganizationalModel()
        for mode, events in rl.groupby([
            'case_type', 'activity_type', 'time_type']):
            om.add_group(set(events['resource']), [mode])

    else:
        raise ValueError

    print('-' * 80)
    measure_values = list()
    
    from orgminer.Evaluation.l2m import conformance
    fitness_score = conformance.fitness(rl, om)
    print('Fitness\t\t= {:.6f}'.format(fitness_score))
    measure_values.append(fitness_score)
    print()

    precision_score = conformance.precision(rl, om)
    print('Precision\t= {:.6f}'.format(precision_score))
    measure_values.append(precision_score)
    print()

    # save the mined organizational model to a file
    with open(fnout_org_model, 'w', encoding='utf-8') as fout:
        om.to_file_csv(fout)
    print('\n[Org. model of {} resources in {} groups exported to "{}"]'
        .format(len(om.resources), om.group_number, fnout_org_model))

