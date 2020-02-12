#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]

if __name__ == '__main__':
    # read event log as input
    # specify the additional attributes included in each input event log
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    # analysis on the input log
    from numpy import mean, std, median
    from datetime import datetime, timedelta
    from collections import defaultdict
    n_events = len(el)
    n_cases = len(el.groupby('case_id'))
    n_activities = len(el.groupby('activity'))
    n_resources = len(el.groupby('resource'))

    '''
    import re
    regex = re.compile(r'_\d\d\d')
    code_set = set()
    for code in set(el['action_code']):
        match = regex.search(code)
        if match is not None:
            code_set.add(code[:match.start()])
        else:
            exit('[Error] Non-matching action code')

    print('total# of codes: {}'.format(len(code_set)))
    print(code_set)
    '''


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
    counter_dist_num_res_case = defaultdict(lambda: 0)
    for case_id, trace in el.groupby('case_id'):
        case_start = datetime.strptime(sorted(trace['timestamp'])[0], fmt)
        case_end = datetime.strptime(sorted(trace['timestamp'])[-1], fmt)
        case_durations.append((case_end - case_start).seconds)
        counter_dist_num_res_case[len(set(trace['resource']))] += 1
    mean_duration_per_case = mean(case_durations)
    std_duration_per_case = std(case_durations)

    # show the result of analysis
    # Overview
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
    # Case perspective
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

    # Resource perspective
    print()
    print('-' * 35 + 'Resource perspective' + '-' * 35)
    print('mean activities per resource = \t\t\t{:.1f} (std = {:.1f})'.format(
        mean_activities_per_resource, std_activities_per_resource))
    print('distribution of number of resources per case:')
    for num_res in sorted(counter_dist_num_res_case.keys()):
        print('\tCase handled by {} resources:\t{} ({:.0%})'.format(
            num_res, counter_dist_num_res_case[num_res], 
            counter_dist_num_res_case[num_res] / n_cases))

    # Time perspective
    print()
    print('-' * 35 + 'Time perspective' + '-' * 35)
    print('mean duration per case = \t\t\t{:.1f} seconds (std = {:.1f})'.format(
        mean_duration_per_case, std_duration_per_case))

