#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

fn_event_log = sys.argv[1]
fnout_social_network = sys.argv[2]
#mining_option = sys.argv[3]
#additional_params = sys.argv[4:] if len(sys.argv) > 4 else None

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)

    print('Input a number to choose a solution:')
    print('\t0. Metrics based on possible causality (van der Aalst)')
    print('\t1. Metrics based on joint cases (van der Aalst)')
    print('\t2. Metrics based on joint activities (van der Aalst)')
    print('\t3. Metrics based on special event types (van der Aalst)')
    print('Option: ', end='')
    mining_option = int(input())


    if mining_option in [3]:
        print('Warning: These options are closed for now. Activate them when necessary.')
        exit(1)
    elif mining_option == 0:
        print('For the options below, refer to van der Aalst\'s CSCW 2005 paper.')
        print('Consider "Direct succession" Y/n? YES by default: ', end='')
        opt_CD = input()
        if opt_CD in ['', 'y', 'Y']:
            cds = True
        elif opt_CD in ['n', 'N']:
            cds = False
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

        print('Consider "Multiple transfers" Y/n? YES by default: ', end='')
        opt_CM = input()
        if opt_CM in ['', 'y', 'Y']:
            cmt = True
        elif opt_CM in ['n', 'N']:
            cmt = False
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

        print('Ignore real Causality dependencies by default.')

        from orgminer.SocialNetworkMiner import causality

        print('Input a number to specify a metric:')
        print('\t0. Handover of work')
        print('\t1. Subcontracting')
        metric_option = int(input())

        if metric_option in [1]:
            print('Warning: These options are closed for now. Activate them when necessary.')
            exit(1)
        elif metric_option == 0:
            if not cds:
                print('Warning: These options are closed for now. Activate them when necessary.')
                exit(1)
            sn = causality.handover(el, real_causality=False,
                    direct_succession=cds, multiple_transfers=cmt)
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    elif mining_option == 1:
        from orgminer.SocialNetworkMiner import joint_cases 
        sn = joint_cases.working_together(el)

    elif mining_option == 2:
        from orgminer.SocialNetworkMiner import joint_activities
        print('Should log scale be used? y/N NO by default: ', end='')
        opt_ls = input()
        if opt_ls in ['y', 'Y']:
            ls = True
        elif opt_ls in ['', 'n', 'N']:
            ls = False
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

        profiles = joint_activities.performer_activity_matrix(
            el, use_log_scale=ls)

        print('Input a number to specify a metric:')
        print('\t0. Distance: Euclidean distance')
        print('\t1. Distance: Manhattan distance')
        print('\t2. Distance: Hamming distance')
        print('\t3. Correlation: Pearson Correlation Coefficient')
        metric_option = int(input())

        if metric_option in [0, 1, 2]: # distance metrics
            distance_metrics = [
                    'euclidean',
                    'cityblock',
                    'hamming']
            sn = joint_activities.distance(
                    profiles, metric=distance_metrics[metric_option])
        elif metric_option in [3]: # correlation metrics
            correlation_metrics = [
                    'pearson']
            sn = joint_activities.correlation(
                    profiles, metric=correlation_metrics[metric_option - 3])
        else:
            raise Exception('Failed to recognize input option!')
            exit(1)

    else:
        raise Exception('Failed to recognize input option!')
        exit(1)

    # save the mined social network to a file
    from networkx import write_gexf
    write_gexf(sn, fnout_social_network)

