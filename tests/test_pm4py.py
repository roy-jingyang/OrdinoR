#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('./')

# List input parameters from shell
fn_event_log = sys.argv[1]
fn_event_log_xes = sys.argv[2]
fn_org_model = sys.argv[3]
fnout = sys.argv[4]

# TODO: suppress the tricky feature of Disco appending transition types after activity labels
def trim_activity_label(s, patt):
    from re import search as regex_search
    match = regex_search(patt, s)
    return s[:match.start()]

if __name__ == '__main__':
    from pm4py.objects.log.importer.xes import factory as xes_import_factory
    log = xes_import_factory.apply(fn_event_log_xes)

    #from pm4py.algo.discovery.heuristics import factory as heuristics_miner
    #net, im, fm = heuristics_miner.apply(log)
    from pm4py.algo.discovery.dfg import factory as dfg_miner
    dfg = dfg_miner.apply(log)

    #from pm4py.visualization.petrinet import factory as pn_vis_factory
    #gviz = vis_factory.apply(net, im, fm)
    from pm4py.visualization.dfg import factory as dfg_vis_factory
    gviz = dfg_vis_factory.apply(dfg, log=log, variant="frequency")

    import pygraphviz as pgv
    graph = pgv.AGraph(gviz.source)

    # TODO: optional, modify the original graph
    for edge in graph.edges_iter():
        edge.attr['label'] = ''

    activity_nodes = dict()
    for node in graph.nodes_iter():
        if node.attr['shape'] == 'box' and node.attr['label'] != '':
            activity_nodes[trim_activity_label(node.attr['label'], r' \(\d+\)')] = node
            # TODO: optional, modify the original graph
            node.attr['label'] = trim_activity_label(
                node.attr['label'], r' \(\d+\)')

    from orgminer.IO.reader import read_disco_csv
    with open(fn_event_log, 'r') as f:
        el = read_disco_csv(f)
        #el = read_disco_csv(f, mapping={'(case) channel': 6})

    from orgminer.ExecutionModeMiner.direct_groupby import ATonlyMiner
    from orgminer.ExecutionModeMiner.direct_groupby import ATCTMiner
    from orgminer.ExecutionModeMiner.direct_groupby import FullMiner
    from orgminer.ExecutionModeMiner.informed_groupby import TraceClusteringFullMiner

    mode_miner = ATonlyMiner(el)
    #mode_miner = FullMiner(el, 
    #    case_attr_name='(case) channel', resolution='weekday')
    #mode_miner = TraceClusteringFullMiner(el,
    #    fn_partition='input/extra_knowledge/bpic12.bosek5.tcreport', resolution='weekday')

    from orgminer.OrganizationalModelMiner.base import OrganizationalModel
    with open(fn_org_model, 'r') as f:
        om = OrganizationalModel.from_file_csv(f)
    
    for og_id, og in om.find_all_groups():
        # add an entity representing the organizational group
        graph.add_node('Group {}'.format(og_id), 
            label='Group {}'.format(og_id),
            color='grey', shape='egg', fillcolor='cadetblue1')
         
        # parsing execution modes
        exec_modes = om.find_group_execution_modes(og_id)[:5]

        for em in exec_modes:
            ct, at, tt = em[0], em[1], em[2]

            # 1. connect groups with process models
            # TODO: suppress the tricky feature of Disco appending transition types after activity labels
            for value in mode_miner.get_values_by_type(at):
                graph.add_edge(
                    'Group {}'.format(og_id), 
                    activity_nodes[trim_activity_label(value, r'-complete')],
                    label=tt)

            # 2. connect groups with case types
            # add case type entities
            if ct in graph.nodes():
                pass
            else:
                graph.add_node(ct,
                    label=ct,
                    color='gold', shape='folder', fillcolor='gold1')
            graph.add_edge('Group {}'.format(og_id), ct,
                label=tt)

    
    from graphviz import Source
    src = Source(graph.string(), format='pdf', engine='dot')
    src.render(filename=fnout, cleanup=True, format='pdf')

