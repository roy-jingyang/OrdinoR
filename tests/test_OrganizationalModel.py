#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

# import methods to be tested below
from orgminer.OrganizationalModelMiner.base import OrganizationalModel

# List input parameters from shell
fn_om = sys.argv[1]


if __name__ == '__main__':
    with open(fn_om, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)
    
    print(om.group_number)
    print(len(om.resources))

