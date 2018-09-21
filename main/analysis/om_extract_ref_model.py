#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout_ref_org_model = sys.argv[2]

if __name__ == '__main__':
    from IO.reader import read_disco_csv
    mapping = {
        'case_id': 0,
        'activity': 1,
        'resource': 2,
        'timestamp': 3,
        #'org:group': -1 # WABO
        #'org:group': 10 # BPiC 2013 Open
        'org:group': 9 # BPiC 2013 Closed
    }

    cases = read_disco_csv(fn_event_log, mapping=mapping)
    # group: 'org:group' => [involved originators]
    og = dict()
    for group_name, events in cases.groupby('org:group'):
        og[group_name] = set(events['resource'])
    print('{} (projected) official groups extracted.'.format(len(og)))

    '''
    # eliminate sub-groups
    group_ids = sorted(og.keys())
    sub_group_ids = set()
    for i in range(len(group_ids) - 1):
        group_i = og[group_ids[i]]
        for j in range(i + 1, len(group_ids)):
            group_j = og[group_ids[j]]

            if group_i <= group_j: # subset
                sub_group_ids.add(group_ids[i])

            if group_j < group_i:
                sub_group_ids.add(group_ids[j])

    print('{}: {} eliminated'.format(sub_group_ids, len(sub_group_ids)))
    for sg_id in sub_group_ids:
        del og[sg_id]

    # check overlapping
    size = 0
    resources = set()
    for gid in sorted(og.keys()):
        print('#resource in {}:\t{}'.format(gid, len(og[gid])))
        for r in og[gid]:
            resources.add(r)
        size += len(og[gid])
    print('{} groups'.format(len(og)))
    print('Total# of resources:\t{}'.format(len(resources)))
    print('Sum of sizes:\t{}'.format(size))
    print('Overlapping:\t', end='')
    print(len(resources) != size)
    '''

    from OrganizationalModelMiner.mining import mode_assignment
    a = mode_assignment.entity_assignment(og, cases)

    from IO.writer import write_om_csv
    write_om_csv(fnout_ref_org_model, og, a)

