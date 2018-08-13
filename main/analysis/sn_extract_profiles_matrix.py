#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./src/')

fn_event_log = sys.argv[1]
fnout_profiles_matrix = sys.argv[2]

if __name__ == '__main__':
    from IO.reader import read_disco_csv
    cases = read_disco_csv(fn_event_log)

    from SocialNetworkMiner.mining.joint_activities import build_performer_activity_matrix
    profiles = build_performer_activity_matrix(cases, use_log_scale=False)

    profiles.to_csv(fnout_profiles_matrix)

