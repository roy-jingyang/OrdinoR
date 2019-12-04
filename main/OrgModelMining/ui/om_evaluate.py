#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

fn_event_log = sys.argv[1]
fn_model = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={'(case) LoanGoal': 7})

    # learn execution modes and convert to resource log
    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import CTonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
    #mode_miner = ATonlyMiner(el)
    mode_miner = ATCTMiner(el, case_attr_name='(case) LoanGoal')

    rl = mode_miner.derive_resource_log(el)

    # read organizational model as input
    from orgminer.OrganizationalModelMiner.base import OrganizationalModel
    with open(fn_model, 'r', encoding='utf-8') as f:
        om = OrganizationalModel.from_file_csv(f)

    from orgminer.Evaluation.l2m import conformance
    measure_values = list()
    print('-' * 80)
    '''
    # TODO: debugging use
    ogs = [og for og_id, og in om.find_all_groups()]
    print('Pct. of solo group\t= {:.1%}'.format(
        sum(1 for og in ogs if len(og) == 1) / len(ogs)))
    '''
    print('-' * 80)
    fitness_score = conformance.fitness(rl, om)
    print('Fitness\t\t= {:.6f}'.format(fitness_score))
    measure_values.append(fitness_score)
    print()
    rc_measure_score = conformance.rc_measure(rl, om)
    print('rc-measure\t= {:.6f}'.format(rc_measure_score))
    measure_values.append(rc_measure_score)
    print()
    precision_score = conformance.precision(rl, om)
    print('Prec. (new2)\t= {:.6f}'.format(precision_score))
    measure_values.append(precision_score)
    print()

    # Overlapping Density & Overlapping Diversity (avg.)
    resources = om.resources
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

