#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
from re import search as regex_search

fn_event_log = sys.argv[1] # bpic15-*
fnout_event_log = sys.argv[2]

if __name__ == '__main__':
    # read event log as input
    from orgminer.IO.reader import read_disco_csv
    # TODO
    with open(fn_event_log, 'r', encoding='utf-8') as f:
        el = read_disco_csv(f, mapping={
            'action_code': 18,
            '(case) last_phase': 14}) # bpic15-*

    patt = r'_\d\d\d\w*'
    def decode(row):
        match = regex_search(patt, row['action_code'])
        if match is None:
            row['subprocess'] = ''
            row['phase'] = ''
            print('No match for row {}'.format(row.name))
        else:
            row['subprocess'] = row['action_code'][:match.start()]
            row['phase'] = row['action_code'][:match.start()+2]
        return row

    el.apply(decode, axis=1).to_csv(fnout_event_log, index=False)

