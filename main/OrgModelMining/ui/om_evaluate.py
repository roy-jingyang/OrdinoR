#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fn_model = sys.argv[2]

def jaccard_index(set_a, set_b):
    return len(set.intersection(set_a, set_b)) / len(set.union(set_a, set_b))

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # learn execution modes and convert to resource log
    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
    from orgminer.ExecutionModeMiner.informed_groupby import \
        TraceClusteringFullMiner
    #mode_miner = ATonlyMiner(el)
    mode_miner = FullMiner(
        el, case_attr_name='(case) channel', resolution='weekday')

    rl = mode_miner.derive_resource_log(el)

    # read organizational model as input
    from orgminer.OrganizationalModelMiner.base import OrganizationalModel
    with open(fn_model, 'r', encoding='utf-8') as f:
        om = OrganizationalModel.from_file_csv(f)

    # Evaluation
    # Global conformance measures
    from orgminer.Evaluation.l2m import conformance
    print('-' * 80)
    fitness_score = conformance.fitness(rl, om)
    print('Fitness\t\t= {:.3f}'.format(fitness_score))
    print()
    precision_score = conformance.precision(rl, om)
    print('Precision\t= {:.3f}'.format(precision_score))
    print()

    jac_resource_sets = jaccard_index(set(rl['resource']), set(om.resources))
    print('Jac. index (resources)\t= {:.3f}'.format(jac_resource_sets))
    print()
    jac_mode_sets = jaccard_index(
        set(rl[['case_type', 'activity_type', 'time_type']]
            .drop_duplicates().itertuples(index=False, name=None)),
        set(om.find_all_execution_modes()))
    print('Jac. index (modes)\t= {:.3f}'.format(jac_mode_sets))
    print()

