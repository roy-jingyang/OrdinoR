#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# filter-sample.py
# Author: Jing Yang
# Purpose:
#   Adopt the BPIC(2014) dataset, to examine whether different settings of 
#   resource assignment on the same pairs of consective tasks (found in a
#   number of different cases) may impact the duration time of performing the
#   tasks.
# Method:
#   Given a pair of consective tasks t1, t2, assume resources r, r', r'' were
#   assigned respectively as (t1->r, t2->r') and (t1->r, t2->r''). Collect all
#   cases where such assignment settings applied, namely Ca for t2->r' and Cb
#   for t2->r''. Either Ca and Cb contains info on the duration of performing
#   task t2. Test if the two distributions of duration time were similar.
# Keyword:
#   resource compatibility, normal distribution, t-test

import sys
import csv
from collections import defaultdict

# Check for duplicated rows
s = set()
ds = list()
with open(sys.argv[1], 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        st = ','.join([x for x in row])
        if st not in s:
            s.add(st)
            ds.append(row)
print('Total # of rows after removing duplicates: {}'.format(len(ds)))

# {(t1, t2, r1) => {r2 => [duration time]}}
d = defaultdict(lambda: defaultdict(lambda: list()))
for row in ds:
    k = tuple(list(int(x) for x in row[:3]))
    r = int(row[3])
    dur = int(row[-1])
    d[k][r].append(dur)

# We want those with:
#   count(r2) > 1 so we can compare between distributions;
#   length([duration time]) > N so we have enough samples.

# It seems that we are not able to perform paired t-test, since the numbers of
# samples from different distributions are not likely to be equal.

with open(sys.argv[2], 'w') as fout:
    for k, r_dur in d.items():
        if len(r_dur.keys()) > 1:
            for r in r_dur.keys():
                fout.write(str(k) + '|' + str(r) + '|' + \
                        ','.join([str(dur) for dur in r_dur[r]]))
                fout.write('\n')
            fout.write('@@\n')

