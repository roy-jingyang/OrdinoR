#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fn_model = sys.argv[2]
fnout_diagnostic_report = sys.argv[3]
opt_repair = sys.argv[4] # {'repair', 'reveal'}
threshold_repair = float(sys.argv[5]) if opt_repair == 'repair' else None

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
    #mode_miner = TraceClusteringFullMiner(
    #    el, fn_partition='./input/extra_knowledge/wabo.bosek5.tcreport',
    #    resolution='weekday')

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
    from collections import defaultdict
    mode_occurrence = rl.groupby([
        'case_type', 'activity_type', 'time_type']).size().to_dict()
    group_size = dict()

    from orgminer.Evaluation.l2m.diagnostics import \
        group_relative_focus, group_relative_stake, member_coverage
    from scipy.stats import hmean
    scores_rel_focus = defaultdict(lambda: dict())
    scores_rel_stake = defaultdict(lambda: dict())
    scores_coverage = defaultdict(lambda: dict())

    for og_id, og in om.find_all_groups():
        if og_id == 3:
            # do something
            from orgminer.Evaluation.l2m.diagnostics import \
                member_mode_contribution
            dist = member_mode_contribution(
                og, 
                ('CT.Internet', 
                'AT.T04 Determine confirmation of receipt-complete', 
                'TT.6'),
                rl
            )
            print(dist)
        else:
            continue

        '''
        group_size[og_id] = len(og)
        for cap in om.find_group_execution_modes(og_id):
            scores_rel_focus[cap][og_id] = group_relative_focus(og, cap, rl)
            scores_rel_stake[cap][og_id] = group_relative_stake(og, cap, rl)
            scores_coverage[cap][og_id] = member_coverage(og, cap, rl)
        '''
    
    exit()

    print()
    diag_results = list()
    for mode, val in scores_coverage.items():
        for og_id, coverage in val.items():
            rel_focus = scores_rel_focus[mode][og_id]
            rel_stake = scores_rel_stake[mode][og_id]

            mf_scaled = mode_occurrence[mode] / max(mode_occurrence.values())
            gs_scaled = group_size[og_id] / max(group_size.values())

            impact_index = mf_scaled * gs_scaled / coverage
            diag_results.append([
                mode, og_id,
                mode_occurrence[mode],
                mf_scaled, 
                group_size[og_id], 
                gs_scaled,
                coverage,
                rel_focus,
                rel_stake,
                impact_index
            ])

    diag_results.sort(key=lambda row: row[-1], reverse=True)

    # Calculate and export model repairing results
    with open(fnout_diagnostic_report, 'w+') as fout:
        if opt_repair == 'repair':
            fout.write(
                'mode;group;mode-freq;m-f scaled;group-size;g-s scaled;' +
                'coverage;rel_focus;rel_stake;impact_index;' +
                'fitness;precision' + '\n')
        elif opt_repair == 'reveal':
            fout.write(
                'mode;group;mode-freq;m-f scaled;group-size;g-s scaled;' +
                'coverage;rel_focus;rel_stake;impact_index' + '\n')

        else:
            exit('Error')

        count = 0
        for row in diag_results:
            count += 1

            mode = row[0]
            og_id = row[1]

            if opt_repair == 'repair':
                if count > len(diag_results) * threshold_repair:
                    break

                # NOTE: directly operating private members of a model instance
                om._cap[og_id].remove(mode)
                om._rcap[mode].remove(og_id)

                fitness_score_new = conformance.fitness(rl, om)
                precision_score_new = conformance.precision(rl, om)

                fout.write(
                    ';'.join(list(str(v) for v in row)) + ';'
                    '{:.3f};{:.3f}'.format(
                        fitness_score_new, precision_score_new) 
                    + '\n')
            elif opt_repair == 'reveal':
                fout.write(
                    ';'.join(list(str(v) for v in row))
                    + '\n')
            else:
                exit('Error')

