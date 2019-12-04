#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# import methods to be tested below
from orgminer.Preprocessing.log_augmentation import append_case_duration

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    from orgminer.IO.reader import read_disco_csv
    with open(filename_input, 'r') as f:
        #el = read_disco_csv(f)
        el = read_disco_csv(f, mapping={
            'action_code': 18,
            '(case) last_phase': 14}) # bpic15-*

    print(el)
    el = append_case_duration(el)
    print(el)
