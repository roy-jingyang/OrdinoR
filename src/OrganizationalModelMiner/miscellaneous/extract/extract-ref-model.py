#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv
import networkx as nx
from collections import defaultdict

f_event_log = sys.argv[1]
fout_org_model = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    # assign according to 'org:group' field

    model = defaultdict(lambda: set())
    performer_task = defaultdict(lambda: set())
    with open(f_event_log, 'r') as f:
        is_header_line = True
        cnt_events = 0
        for row in csv.reader(f):
            if is_header_line:
                is_header_line = False
            else:
                case_id = row[0]
                activity = row[1]
                resource = row[2]
                #org_group = row[-1] # wabo
                #org_group = row[10] # bpic2013_open
                org_group = row[9] # bpic2013_closed
                model[org_group].add(resource)
                performer_task[resource].add(activity)
    
    '''
    # eliminate sub-groups
    group_ids = sorted(model.keys())
    sub_group_ids = set()
    for i in range(len(group_ids) - 1):
        group_i = model[group_ids[i]]
        for j in range(i + 1, len(group_ids)):
            group_j = model[group_ids[j]]

            if group_i <= group_j: # subset
                sub_group_ids.add(group_ids[i])

            if group_j < group_i:
                sub_group_ids.add(group_ids[j])

    print('{}: {} eliminated'.format(sub_group_ids, len(sub_group_ids)))
    for sg_id in sub_group_ids:
        del model[sg_id]
    '''

    # check overlapping
    size = 0
    resources = set()
    for entity_id in sorted(model.keys()):
        print('#resource in {}:\t{}'.format(entity_id, len(model[entity_id])))
        for r in model[entity_id]:
            resources.add(r)
        size += len(model[entity_id])
    print('{} groups'.format(len(model)))
    print('Total# of resources:\t{}'.format(len(resources)))
    print('Sum of sizes:\t{}'.format(size))
    print('Overlapping:\t', end='')
    print(len(resources) != size)

    # output to file
    with open(fout_org_model, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['entity_id', 'tasks', 'resources'])
        for entity_id in model.keys():
            entity_assignment = set()
            for r in model[entity_id]:
                for a in performer_task[r]:
                    entity_assignment.add(a)
            writer.writerow([
                'NULL' if entity_id == '' else entity_id,
                ';'.join(a for a in entity_assignment),
                ';'.join(r for r in model[entity_id])])

