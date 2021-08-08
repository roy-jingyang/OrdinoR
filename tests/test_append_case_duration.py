#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# import methods to be tested below
from ordinor.io import read_disco_csv
from ordinor.utils.log_preprocessing import append_case_duration

# List input parameters from shell
filename_input = sys.argv[1]

if __name__ == '__main__':
    el = read_disco_csv(filename_input)

    print(el)
    el = append_case_duration(el)
    print(el)
