#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

if __name__ == '__main__':
    # Derived resource log RL
    from pandas import DataFrame
    from numpy import array
    rl = DataFrame(array([
        ['Pete', 'normal', 'register', 'afternoon'],
        ['Pete', 'normal', 'register', 'afternoon'],
        ['Ann', 'normal', 'contact', 'afternoon'],
        ['John', 'normal', 'check', 'morning'],
        ['Sue', 'normal', 'check', 'morning'],
        ['Bob', 'VIP', 'register', 'morning'],
        ['John', 'normal', 'decide', 'morning'],
        ['Sue', 'normal', 'decide', 'morning'],
        ['Mary', 'VIP', 'check', 'afternoon'],
        ['Mary', 'VIP', 'decide', 'afternoon']]),
        columns=['resource', 'case_type', 'activity_type', 'time_type']
    )

    # Organizational model OM
    from orgminer.OrganizationalModelMiner.base import OrganizationalModel
    om = OrganizationalModel()
    groups = [
        {'Pete', 'Bob'},    # the registering team
        {'John', 'Sue'},    # the processing team (for normal orders)
        {'Mary'},           # the processing team (for VIP orders)
        {'Ann'}             # the contact team
    ]
    capabilities = [
        [('normal', 'register', 'afternoon'), ('VIP', 'register', 'morning')],
        [('normal', 'check', 'morning'), ('normal', 'decide', 'morning')],
        [('VIP', 'check', 'afternoon'), ('VIP', 'decide', 'afternoon')],
        [('normal', 'contact', 'afternoon')]
    ]
    for i in range(len(groups)):
        om.add_group(groups[i], capabilities[i])

    # Export the toy model
    with open('toy_example.om', 'w+') as fout:
        om.to_file_csv(fout)

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

    # Local diagnostics
    from orgminer.Evaluation.l2m import diagnostics
    for og_id, og in om.find_all_groups():
        index = groups.index(og)
        for mode in capabilities[index]:
            print('Group {}:\t[{}]\t --- {}'.format(
                og_id, ','.join(og), mode))
            print('\tRelFocus = {:.3f}'.format(
                diagnostics.group_relative_focus(og, mode, rl)))
            print('\tRelStake = {:.3f}'.format(
                diagnostics.group_relative_stake(og, mode, rl)))
            print('\tCov = {:.3f}'.format(
                diagnostics.member_coverage(og, mode, rl)))

