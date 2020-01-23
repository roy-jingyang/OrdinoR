#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

fn_event_log = sys.argv[1]
fn_model = sys.argv[2]
fnout_diagnostics_report = sys.argv[3]

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
    mode_miner = ATonlyMiner(el)
    #mode_miner = FullMiner(
    #    el, case_attr_name='(case) channel', resolution='weekday')

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
    print('(Original) Fitness\t= {:.3f}'.format(fitness_score))
    print()
    precision_score = conformance.precision(rl, om)
    print('(Original) Precision\t= {:.3f}'.format(precision_score))
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

    # Local diagnostics (targeting low precision)
    mode_occurrence = rl.groupby([
        'case_type', 'activity_type', 'time_type']).size().to_dict()
    from orgminer.Evaluation.l2m.diagnostics import member_coverage
    from collections import defaultdict
    mode_coverage = defaultdict(lambda: dict())
    group_size = dict()
    for og_id, og in om.find_all_groups():
        group_size[og_id] = len(og)
        for cap in om.find_group_execution_modes(og_id):
            mode_coverage[cap][og_id] = member_coverage(og, cap, rl)
    
    print()
    diag_results = list()
    for mode, val in mode_coverage.items():
        for og_id, coverage in val.items():
            mf_scaled = mode_occurrence[mode] / max(mode_occurrence.values())
            gs_scaled = group_size[og_id] / max(group_size.values())
            impact_index = mf_scaled * gs_scaled / coverage
            diag_results.append([
                mode, og_id,
                mode_occurrence[mode],
                mf_scaled, 
                group_size[og_id], 
                gs_scaled,
                coverage, impact_index
            ])

    diag_results.sort(key=lambda row: row[-1], reverse=True)

    # Calculate and export model repairing results
    with open(fnout_diagnostics_report, 'w+') as fout:
        fout.write(
            'mode;group;mode-freq;m-f scaled;group-size;g-s scaled;' +
            'coverage;impact-index;fitness;precision' + '\n')
        for row in diag_results:
            mode = row[0]
            og_id = row[1]
            # NOTE: directly operating private members of a model instance
            om._cap[og_id].remove(mode)
            om._rcap[mode].remove(og_id)

            fitness_score_new = conformance.fitness(rl, om)
            precision_score_new = conformance.precision(rl, om)
            fout.write(
                ';'.join(list(str(v) for v in row)) + ';'
                '{:.3f};{:.3f}'.format(fitness_score_new, precision_score_new) 
                + '\n')

