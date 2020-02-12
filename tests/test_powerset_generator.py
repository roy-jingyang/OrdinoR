#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from orgminer.OrganizationalModelMiner.utilities import powerset_exclude_headtail

if __name__ == '__main__':
    s = frozenset(range(2, 13))
    it = powerset_exclude_headtail(s, reverse=True, depth=1)
    for x in it:
        print(x)

