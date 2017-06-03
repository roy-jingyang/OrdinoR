#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# hypo-test.py
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
#   resource compatibility, normal distribution, hypothesis testing

import sys
from collections import defaultdict
from scipy.stats import ttest_ind, t

if __name__ == '__main__':
    ds = defaultdict(lambda: defaultdict(lambda: None))
    # Load samples from data file
    # {(t1, t2, r1) => {r2 => [duration time]}}
    with open(sys.argv[1], 'r') as f:
        while True:
            line = f.readline()
            if not line.startswith('#'):
                if line == '':
                    break
                else:
                    line = line.strip()
                    k_r2_dis = line.split('|')
                    t1_t2_r1 = tuple(int(x) 
                            for x in k_r2_dis[0][1:-1].split(','))
                    r2 = int(k_r2_dis[1])
                    dis = [int(x) for x in k_r2_dis[2].split(',')]
                    ds[t1_t2_r1][r2] = dis
    
    # Perform t-test on two independent distributions, assume equal var
    # significance level
    alpha = 0.05

    group_cnt = 0
    for k in ds.keys():
        group_cnt += 1
        print('-' * 79)
        print('Group {} - Examine for settings t1={}, t2={}, r1={}:'.format(
            group_cnt, k[0], k[1], k[2]), end='')
        r2s = list(ds[k].keys())
        print(' {} distributions of duration time of t2'.format(len(r2s)) + 
                ' (# of possible values that r2 takes.)')
        pair_cnt = 0
        statistics = list()
        pvalues = list()
        for i in range(len(r2s)):
            for j in range(i + 1, len(r2s)):
                pair_cnt += 1
                str_a = '({}, res={})^({}, res={})'.format(k[0], k[2], k[1],
                        r2s[i])
                str_b = '({}, res={})^({}, res={})'.format(k[0], k[2], k[1],
                        r2s[j])
                print('Pair {} - ('.format(pair_cnt)+ str_a + ') vs. (' + str_b 
                        + '): ')
                sample_a = ds[k][r2s[i]]
                sample_b = ds[k][r2s[j]]
                # degree of freedom
                df = len(sample_a) + len(sample_b) - 2
                # Critital value (lower tail)
                cv_lower_t = t.isf(1 - alpha / 2, df)
                test_result = ttest_ind(sample_a, sample_b)
                stat = test_result[0]
                pvalue = test_result[1]
                print('Test statistic: T = {}, with pvalue = {}.'.format(stat,
                    pvalue), end='')
                if abs(stat) > abs(cv_lower_t):
                    print(' [Reject] H0 under significance level {}:'.format(
                        alpha) + ' |T| > {}'.format(abs(cv_lower_t))) 
                else:
                    print(' [Accept] H0 under significance level {}:'.format(
                        alpha) + ' |T| <= {}'.format(abs(cv_lower_t))) 
                print('\n')
                statistics.append(stat)
                pvalues.append(pvalue)
        print('\n')

