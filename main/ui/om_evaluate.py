#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fn_model = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'product': -2})
        #el = read_disco_csv(f, mapping={'(case) channel': 6})

    # learn execution modes and convert to resource log
    from ExecutionModeMiner.naive_miner import ATonlyMiner
    from ExecutionModeMiner.naive_miner import CTonlyMiner
    from ExecutionModeMiner.naive_miner import ATCTMiner
    naive_exec_mode_miner = ATonlyMiner(el)
    #naive_exec_mode_miner = CTonlyMiner(el, case_attr_name='product')
    #naive_exec_mode_miner = ATCTMiner(el, case_attr_name='(case) channel')

    rl = naive_exec_mode_miner.derive_resource_log(el)

    # read organizational model as input
    from OrganizationalModelMiner.base import OrganizationalModel
    with open(fn_model, 'r', encoding='utf-8') as f:
        om = OrganizationalModel.from_file_csv(f)

    measure_values = list()
    print('-' * 80)
    # TODO: debugging use
    ogs = om.find_all_groups()
    print('Pct. of solo group\t= {:.1%}'.format(
        sum(1 for og in ogs if len(og) == 1) / len(ogs)))
    print('-' * 80)
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
    print()
    un_measure_score = conformance.un_measure(rl, om)
    print('un-measure\t= {:.6f}'.format(un_measure_score))
    measure_values.append(un_measure_score)
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

