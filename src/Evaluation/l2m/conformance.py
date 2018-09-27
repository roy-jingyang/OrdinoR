# -*- coding: utf-8 -*-

'''
This module contains the implmentation of the conformance checking measures
proposed in the OrgMining 2.0 framework.
'''

# Fitness related
def _count_conforming_events(rl, om):
    count = 0
    for res_e in rl.itertuples():
        is_conformed = False
        res = res_e.resource

        e_exec_mode = (res_e.case_type, res_e.activity_type, res_e.time_type)
        cand_groups = om.get_candidate_groups(e_exec_mode)
        for g in cand_groups:
            if res in g:
                is_conformed = True
                break
        
        if is_conformed:
            count += 1
    return count 

def fitness(rl, om):
    n_e_conf = _count_conforming_events(rl, om)
    n_e_res = len(rl)
    return n_e_conf / n_e_res

# Precision related
def precision(rl, om):
    counts = list()
    cand_all = set()
    for res_e in rl.itertuples():
        e_exec_mode = (res_e.case_type, res_e.activity_type, res_e.time_type)
        cand_groups = om.get_candidate_groups(e_exec_mode)

        cand_res = set.union(*cand_groups)
        counts.append(len(cand_res))

        cand_all.update(cand_res)

    n_cand_all = len(cand_all)
    n_e_res = len(rl)
    prec_sum = 0.0
    for n_cand_res in counts:
        prec_sum += (n_cand_all - n_cand_res) / (n_cand_all - 1)

    return prec_sum / n_e_res

