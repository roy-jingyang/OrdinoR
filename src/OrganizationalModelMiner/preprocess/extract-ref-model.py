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
        # BPiC 2013 Volvo Service Desk: Incident Mngt. Syst.
        for line in f:
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                row = line.split(';')
                caseid = row[0] # SR Number
                ctimestamp = row[1] # Change Date+Time
                resource = row[-1]
                activity = row[2] + row[3]
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        # BPiC 2015 Building Permit Application: Municiality 3
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                caseid = row[0] 
                ctimestamp = row[3] # Complete timestamp 
                resource = row[2]
                activity = row[1] # Activity code
                cases[caseid].append((caseid, activity, resource, ctimestamp))
        '''
        # The 'WABO' event log data
        for row in csv.reader(f):
            ln += 1
            if is_header_line:
                is_header_line = False
            else:
                activity = row[1]
                resource = row[2]
                org_group = row[13]
                # aggregate by 'org_group'
                model[org_group].add(resource)
                performer_task[resource].add(activity)
    
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

