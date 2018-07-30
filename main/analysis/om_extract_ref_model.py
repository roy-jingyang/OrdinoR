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
    print('{} ground-truth organizational groups extracted.'.format(len(og)))

    from OrganizationalModelMiner import entity_assignment
    a = entity_assignment.assign(og, cases)

    from IO.writer import write_om_csv
    write_om_csv(fnout_ref_org_model, og, a)

