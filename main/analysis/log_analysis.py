#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import csv

fn_event_log = sys.argv[1]
def read_disco_csv(f, mapping=None, header=True):
    '''
    Params:
        f: file object
            File object of the event log being imported.
        mapping: dict, optional
            A python dictionary that denotes the mapping from CSV column
            numbers to event log attributes.
        header: boolean, optional
            True if the event log file contains a header line, False otherwise.
    Returns:
        el: DataFrame
            The event log in pandas DataFrame form.
    '''

    ld = list()
    is_header_line = True
    line_count = 0

    for row in csv.reader(f):
        line_count += 1
        if is_header_line:
            is_header_line = False
            pass
        else:
            # the default mapping is defined as below
            e = {
                'case_id': row[0],
                'activity': row[1],
                'resource': row[2],
                'timestamp': row[3]
            }
            # add addtional attributes mapping specified
            if mapping is not None:
                for attr, col_num in mapping.items():
                    if attr not in e:
                        e[attr] = row[col_num]

            ld.append(e)

    from pandas import DataFrame
    el = DataFrame(ld)

    print('Imported successfully. {} lines scanned.'.format(line_count))

    return el

if __name__ == '__main__':
    # read event log as input
    # specify the additional attributes included in each input event log

    with open(fn_event_log, 'r', encoding='utf-8') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={'(case) AMOUNT_REQ': -3})

    from collections import defaultdict
    from numpy import mean
    res_amount = dict()
    for r, events in el.groupby('resource'):
        count = len(events['(case) AMOUNT_REQ'])
        avg_amount = events['(case) AMOUNT_REQ'].astype('int').mean()
        std_amount = events['(case) AMOUNT_REQ'].astype('int').std()
        res_amount[r] = (count, avg_amount, std_amount) 

    for r in sorted(res_amount.keys()):
        print('{},{},{},{}'.format(r, 
            res_amount[r][0], res_amount[r][1], res_amount[r][2]))
    '''
    # analysis on the input log
    from numpy import mean, std, median
    from datetime import datetime, timedelta
    n_events = len(el)
    n_cases = len(el.groupby('case_id'))
    n_activities = len(el.groupby('activity'))
    n_resources = len(el.groupby('resource'))

    fmt = '%Y/%m/%d %H:%M:%S.%f'
    start_time = datetime.strptime(sorted(el['timestamp'])[0], fmt)
    end_time = datetime.strptime(sorted(el['timestamp'])[-1], fmt)

    mean_events_per_case = mean(
        [len(group) for name, group in el.groupby('case_id')])
    std_events_per_case = std(
        [len(group) for name, group in el.groupby('case_id')])
    median_events_per_case = median(
        [len(group) for name, group in el.groupby('case_id')])
    
    mean_activities_per_case = mean(
        [len({x for x in group['activity']}) 
            for name, group in el.groupby('case_id')])
    std_activities_per_case = std(
        [len({x for x in group['activity']})
            for name, group in el.groupby('case_id')])
    median_activities_per_case = median(
        [len({x for x in group['activity']})
            for name, group in el.groupby('case_id')])

    mean_resources_per_case = mean(
        [len({x for x in group['resource']}) 
            for name, group in el.groupby('case_id')])
    std_resources_per_case = std(
        [len({x for x in group['resource']})
            for name, group in el.groupby('case_id')])

    mean_activities_per_resource = mean(
        [len({x for x in group['activity']})
            for name, group in el.groupby('resource')])
    std_activities_per_resource = std(
        [len({x for x in group['activity']})
            for name, group in el.groupby('resource')])

    case_durations = list()
    for case_id, trace in el.groupby('case_id'):
        case_start = datetime.strptime(sorted(trace['timestamp'])[0], fmt)
        case_end = datetime.strptime(sorted(trace['timestamp'])[-1], fmt)
        case_durations.append((case_end - case_start).seconds)
    mean_duration_per_case = mean(case_durations)
    std_duration_per_case = std(case_durations)

    # show the result of analysis
    print()
    print('-' * 35 + 'Overview stats' + '-' * 35)
    print('Total number of events = \t\t\t{:.1f}'.format(n_events))
    print('Total number of cases = \t\t\t{:.1f}'.format(n_cases))
    print('Total number of activities = \t\t\t{:.1f}'.format(n_activities))
    print('Total number of resources = \t\t\t{:.1f}'.format(n_resources))
    print('Log start time:\t\t\t{}'.format(start_time.strftime(fmt)))
    print('Log end time:\t\t\t{}'.format(end_time.strftime(fmt)))
    print('Total duration:\t\t\t{:.1f} days'.format(
        (end_time - start_time).days))
    
    print()
    print('-' * 35 + 'Case perspective' + '-' * 35)
    print('mean events per case = \t\t\t{:.1f} (std = {:.1f})'.format(
        mean_events_per_case, std_events_per_case))
    print('median events per case = \t\t\t{:.1f}'.format(median_events_per_case))
    print('mean activities per case = \t\t\t{:.1f} (std = {:.1f})'.format(
        mean_activities_per_case, std_activities_per_case))
    print('median activities per case = \t\t\t{:.1f}'.format(
        median_activities_per_case))
    print('mean resources per case = \t\t\t{:.1f} (std = {:.1f})'.format(
        mean_resources_per_case, std_resources_per_case))

    print()
    print('-' * 35 + 'Resource perspective' + '-' * 35)
    print('mean activities per resource = \t\t\t{:.1f} (std = {:.1f})'.format(
        mean_activities_per_resource, std_activities_per_resource))

    print()
    print('-' * 35 + 'Time perspective' + '-' * 35)
    print('mean duration per case = \t\t\t{:.1f} seconds (std = {:.1f})'.format(
        mean_duration_per_case, std_duration_per_case))
    '''

