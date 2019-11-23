#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

def filter_cases_by_frequency(log, case_attr, threshold=0.1, 
    query_only=False):
    frequent_case_classes = set()
    num_total_cases = len(set(log['case_id']))

    for case_class, events in log.groupby(case_attr):
        num_cases = len(set(events['case_id']))
        if (num_cases / num_total_cases) >= threshold:
            frequent_case_classes.add(case_class)
    
    if query_only:
        return sorted(frequent_case_classes)
    else:
        filtered = log.loc[log[case_attr].isin(frequent_case_classes)]
        print('''[filter_cases_by_frequency] 
            Frequent values of attribute "{}": {}
            {} / {} ({:.1%}) cases are kept.'''.format(
            case_attr, sorted(set(frequent_case_classes)),
            len(set(filtered['case_id'])),
            num_total_cases,
            len(set(filtered['case_id'])) / num_total_cases))
        return filtered

def filter_events_by_frequency(log, event_attr, threshold=0.1, 
    query_only=False):
    frequent_classes = set()
    num_total_events = len(log)

    for class_name, events in log.groupby(event_attr):
        if (len(events) / num_total_events) >= threshold:
            frequent_classes.add(class_name)

    if query_only:
        return sorted(frequent_classes)
    else:
        filtered = log.loc[log[event_attr].isin(frequent_classes)]
        print('''[filter_events_by_frequency] 
            Frequent values of attribute "{}": {}
            {} / {} ({:.1%}) events are kept.'''.format(
            event_attr, sorted(set(frequent_classes)),
            len(filtered),
            num_total_events,
            len(filtered) / num_total_events))
        return filtered

def filter_events_by_active_resources(log, threshold=0.01,
    query_only=False):
    active_resources = set()
    num_total_events = len(log)

    for resource, events in log.groupby('resource'):
        if (len(events) / num_total_events) >= threshold and resource != '':
            active_resources.add(resource)

    if query_only:
        return sorted(active_resources)
    else:
        filtered = log.loc[log['resource'].isin(active_resources)]
        print('''{} resources found active with {} events.\n'''.format(
            len(set(filtered['resource'])), len(filtered)))
        return filtered

