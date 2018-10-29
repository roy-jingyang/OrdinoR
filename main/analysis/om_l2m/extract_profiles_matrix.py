#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout_profiles_matrix = sys.argv[2]

if __name__ == '__main__':
    from IO.reader import read_disco_csv
    cases = read_disco_csv(fn_event_log)

    from ResourceProfiler.raw_profiler import performer_activity_frequency
    profiles = performer_activity_frequency(cases, use_log_scale=False)

    profiles.to_csv(fnout_profiles_matrix)

