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
    with open(f_event_log, 'r', encoding='windows-1252') as f:
        is_header_line = True
        ln = 0
        '''
        # BPiC 2013 Volvo VINST: Problem Mngt. Open problem
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0]
                activity = row[1]
                resource = row[2]
                org_group = row[-1]
                # aggregate by 'org:group'
                if True:
                    model[org_group].add(resource)
                    performer_task[resource].add(activity)
                else:
                    pass

        # BPiC 2013 Volvo VINST: Problem Mngt. Closed problem
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0]
                activity = row[1]
                resource = row[2]
                org_group = row[-1]
                # aggregate by 'org:group'
                if True:
                    model[org_group].add(resource)
                    performer_task[resource].add(activity)
                else:
                    pass
        '''

        # The 'WABO' event log data
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                activity = row[1]
                resource = row[2]
                org_group = row[-1]
                # aggregate by 'org_group'
                if True:
                    model[org_group].add(resource)
                    performer_task[resource].add(activity)
                else:
                    pass
        '''
        '''
    
    # check overlapping
    size = 0
    resources = set()
    for entity_id in model.keys():
        print('#resource in {}:\t{}'.format(entity_id, len(model[entity_id])))
        for r in model[entity_id]:
            resources.add(r)
        size += len(model[entity_id])
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

