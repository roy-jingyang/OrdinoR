#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
#fn_om = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={'(case) LoanGoal': 7})

    # learn execution modes and convert to resource log
    #from ExecutionModeMiner.direct_groupby import ATonlyMiner
    from ExecutionModeMiner.direct_groupby import ATCTMiner
    #naive_exec_mode_miner = ATonlyMiner(el)
    naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')
    rl = naive_exec_mode_miner.derive_resource_log(el)

    from OrganizationalModelMiner.base import OrganizationalModel
    with open(fn_om, 'r', encoding='utf-8') as f:
        om = OrganizationalModelMiner.from_file_csv(f)

    ogs = om.find_all_groups()

    from OrganizationalModelMiner.mode_assignment import assign_by_any as assign
    om_new = OrganizationalModel()
    for og in ogs:
        modes = assign(og, rl)
        om_new.add_group(og, modes)

    from Evaluation.l2m import conformance
    print()
    print('Fitness\t\t= {:.6f}'.format(conformance.fitness(rl, om_new)))
    print('Precision\t= {:.6f}'.format(conformance.precision(rl, om_new)))


