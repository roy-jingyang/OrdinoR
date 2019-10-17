#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below


# List input parameters from shell
fn_event_log = sys.argv[1]
fnout_network = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from IO.reader import read_disco_csv
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f)
    
    print(len(set(el['resource'])))
    from SocialNetworkMiner.joint_cases import working_together
    sn = working_together(el, self_loop=True) 
    from networkx import write_gexf
    sn = write_gexf(sn, fnout_network)
    '''
    from OrganizationalModelMiner.community.graph_partitioning import (
        _mjc)
    for k in range(2, 10):
        ogs, _ = _mjc(el, k, method='centrality')
        if len(ogs) == 0:
            print('Invalid')
        else:
            print('Org. groups discovered with {} groups and {} resources.'.format(
                len(ogs), sum(len(x) for x in ogs)))
    '''

