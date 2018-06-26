#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

# import methods to be tested below
from IO.reader import read_disco_csv
from IO.writer import write_om_csv
from OrganizationalModelMiner.mining import default_mining
from OrganizationalModelMiner import entity_assignment

# List input parameters from shell
filename_event_log = sys.argv[1]
filename_result = sys.argv[2]


if __name__ == '__main__':
    cases = read_disco_csv(filename_event_log)
    og = default_mining.mine(cases)
    assignment = entity_assignment.assign(og, cases)
    write_om_csv(filename_result, og, assignment)

