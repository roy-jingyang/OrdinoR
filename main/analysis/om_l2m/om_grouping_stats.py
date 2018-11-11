#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Build resource social network depending on group membership. 

import sys
sys.path.append('./src/')

from OrganizationalModelMiner.base import OrganizationalModel

fn_org_model = sys.argv[1]

if __name__ == '__main__':
    # read organizational model
    with open(fn_org_model, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)


    n_groups = om.size()
    resources = om.resources()
    n_resources = len(resources)
    groups = om.find_all_groups()

    print('{} resources IN {} groups.'.format(
        n_resources, n_groups))
    print()
    for i, g in enumerate(groups):
        print('Group #{}:\t'.format(i), end='')
        print(sorted(g))
    print()
    print('Avg:\t{:.3f} resources per group;'.format(
        sum(len(g) for g in groups) / n_groups))

    resource_membership = dict()
    for r in resources:
        resource_membership[r] = tuple(sorted(om.find_group_ids(r)))

    max_num_membership = max(len(mem) for mem in resource_membership.values())
    print('Avg:\t{:.3f} groups per resource.'.format(
        sum(len(mem) for mem in resource_membership.values()) / n_resources))
    print()
    for i in range(1, max_num_membership + 1):
        print('#resources with membership of {} groups:\t{}'.format(
            i, 
            sum([1 for r, mem in resource_membership.items() if len(mem) == i])))




