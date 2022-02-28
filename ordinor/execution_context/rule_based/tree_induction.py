from itertools import count
from copy import deepcopy
from multiprocessing import cpu_count, Pool

import pandas as pd
import numpy as np

import ordinor.exceptions as exc
from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const

from ordinor.execution_context.base import BaseMiner

from .AtomicRule import AtomicRule
from .TreeNode import Node
from .score_funcs import dispersal, impurity
from .rule_generators import NumericRuleGenerator, CategoricalRuleGenerator

class TreeInductionMiner(BaseMiner):
    def __init__(self, el, spec, eps):
        # Read event log
        if el is not None:
            el = check_convert_input_log(el)
        self._log = el

        # Parse specification
        self.cand_attr_pool = spec['cand_attrs']
        # TODO: model given rules from the specification

        # Set epsilon
        self.eps = eps

        # Init matrices to represent states and score:
        # event-resource (this should remain constant)
        self._m_event_r = el[const.RESOURCE].copy()
        # event-node ("execution context")
        # all events belong to the same node (root) at init
        self._m_event_node = None
        # node-type (ct, at, tt)
        # empty at the beginning; grow as nodes are created
        self._m_node_t = dict()

        # Init node list
        self._root = None
        self._leaves = list()

        # Init score recorders
        self.val_dis = None
        self.val_imp = None
        self.val_target = None


    def _build_ctypes(self, el, **kwargs):
        # TODO
        pass

    def _build_atypes(self, el, **kwargs):
        # TODO
        pass

    def _build_ttypes(self, el, **kwargs):
        # TODO
        pass
    
    def _init_label_generators(self):
        # Init label generators for nodes creation (except the root "0")
        self._gen_node_label = count(start=1, step=1)
        self._gen_ct_label = count(start=1, step=1)
        self._gen_at_label = count(start=1, step=1)
        self._gen_tt_label = count(start=1, step=1)
    
    def _init_matrices(self):
        self._m_event_node = pd.Series(
            [0] * len(self._m_event_r), 
            index=self._m_event_r.index
        )
        self._m_node_t.clear()
    
    def _get_matrices_copy(self):
        return (
            self._m_event_r.copy(deep=True),
            self._m_event_node.copy(deep=True), 
            deepcopy(self._m_node_t)
        )

    def _init_leaves(self):
        # clear list
        self._leaves.clear()

        # build root node (all labels marked "0")
        # root node contains all events
        # root node is marked with an empty atomic rule
        root = Node(
            label=0, event_ids=self._log.index,
            ct_label=0, at_label=0, tt_label=0,
            step_rule=None, parent_rule=None
        )
        # add root to node list
        self._leaves.append(root)
        # update node matrix
        self._m_node_t[root.label] = {
            'CT': root.ct_label, 'AT': root.at_label, 'TT': root.tt_label
        }
        self._root = root

        # also, init scores as per root node status
        self.val_dis = 0.0
        self.val_imp = impurity(self._m_event_node, self._m_event_r)
        self.val_target = 0.0
        return
    
    def print_scores(self):
        print('Dis. = {:.6f}'.format(self.val_dis), end=', ')
        print('Imp. = {:.6f}'.format(self.val_imp), end=', ')
        print('Target = {:.6f}'.format(self.val_target))
    
    def print_tree(self):
        l_all_nodes = list(self.traverse_tree())

        print('*' * 80)

        print('=' * 30 + ' TREE SUMMARY ' + '=' * 30)
        print(f"Number of nodes:\t{len(l_all_nodes)}")
        print('Current scores:', end='\t')
        self.print_scores()

        print('=' * 30 + ' LEAF NODES ' + '=' * 30)
        print(f"Tree contains {len(self._leaves)} leaf nodes:")
        for node in self._leaves:
            print(node)
        
        print('=' * 25 + ' ENCODING TYPES WITH RULES ' + '=' * 25)
        rules_ct = []
        rules_at = []
        rules_tt = []
        for node in self._leaves:
            rule_ct, rule_at, rule_tt = node.composited_rule.to_types()
            rules_ct.append(rule_ct)
            rules_at.append(rule_at)
            rules_tt.append(rule_tt)
        print('Rules for Case Types:')
        for r in rules_ct:
            print(f"\t{r}")
        print('Rules for Activity Types:')
        for r in rules_at:
            print(f"\t{r}")
        print('Rules for Time Types:')
        for r in rules_tt:
            print(f"\t{r}")


        print('*' * 80)

    def traverse_tree(self):
        l_all_nodes = []

        i = 0
        l_all_nodes.append(self._root)
        while i < len(l_all_nodes):
            curr_node = l_all_nodes[i]
            l_all_nodes.extend(curr_node.children)
            i += 1
        
        for node in l_all_nodes:
            yield node

    def fit_decision_tree(self):
        # main procedure
        # initialize
        self._init_matrices()
        self._init_label_generators()
        self._init_leaves()
        print('Decision tree initialized with empty root node\n', end='\t')
        self.print_scores()

        # iterative tree induction
        print(f'Start to fit decision tree with epsilon = {self.eps}')
        tree_expandable = True
        while tree_expandable:
            next_node, split = self._pop_next_node()

            if next_node is None:
                print('There is no more next node to be split')
                tree_expandable = False
                print('Tree fitted. Procedure stops with final scores:')
            elif np.abs(split['target'] - self.val_target) < self.eps:
                print('Change to target value is insignificant')
                tree_expandable = False
                self._leaves.append(next_node)
                print('Tree fitted. Procedure stops with final scores:')
            else:
                print(f"Tree grows by splitting node [{next_node.label}] on attribute `{split['attr']}`")

                # locate the subset of events held by the node
                log = self._log.loc[next_node.event_ids]

                # apply the split (rules) found and create child nodes
                # one child node per split rule
                for rule in split['split_rules']:
                    child_node_label = next(self._gen_node_label)

                    # set (new) type labels depending on the rule applied
                    # by default type labels are inherited from the parent 
                    new_ct_label = next_node.ct_label
                    new_at_label = next_node.at_label
                    new_tt_label = next_node.tt_label
                    # update a dimension, depending on the rule applied
                    if split['attr_dim'] == 'CT':
                        new_ct_label = next(self._gen_ct_label)
                    elif split['attr_dim'] == 'AT':
                        new_at_label = next(self._gen_at_label)
                    elif split['attr_dim'] == 'TT':
                        new_tt_label = next(self._gen_tt_label)
                    else:
                        raise ValueError
                    
                    # update matrix: node-type
                    self._m_node_t[child_node_label] = {
                        'CT': new_ct_label, 
                        'AT': new_at_label, 
                        'TT': new_tt_label
                    }

                    # locate the events to be assigned to the child node
                    par = rule.apply(log, index_only=True)
                    # update matrix: event-node (assign events to child node)
                    self._m_event_node.loc[par] = child_node_label
                    
                    # create child node
                    child_node = Node(
                        label=child_node_label,
                        event_ids=par,
                        ct_label=new_ct_label, at_label=new_at_label, tt_label=new_tt_label,
                        step_rule=rule,
                        parent_rule=next_node.composited_rule
                    )
                    # attach to the tree
                    next_node.append_child(child_node)
                    # add the child node to queue (update the frontiers)
                    self._leaves.append(child_node)

                # remove the used node from the leaves (update the frontiers)
                del self._m_node_t[next_node.label]

                # update scores
                self.val_dis += split['delta_dis']
                self.val_imp += split['delta_imp']
                self.val_target = split['target']

            print('\t', end='')
            self.print_scores()
        # print tree
        self.print_tree()

    def _evaluate_split(self, node, cand_rules, dim, log):
        # get deep copies for evaluation
        cand_m_event_r, cand_m_event_node, cand_m_node_t = self._get_matrices_copy()
        # evaluation block
        for rule in cand_rules:      
            par = rule.apply(log, index_only=True)

            # subset events
            next_node_label = next(self._gen_node_label)
            cand_m_event_node.loc[par] = next_node_label
            
            # inherit from parent node
            cand_m_node_t[next_node_label] = {
                'CT': node.ct_label, 'AT': node.at_label, 'TT': node.tt_label
            }
            
            # change depending on the rule applied
            if dim == 'CT':
                next_ct_label = next(self._gen_ct_label)
                cand_m_node_t[next_node_label]['CT'] = next_ct_label
            elif dim == 'AT':
                next_at_label = next(self._gen_at_label)
                cand_m_node_t[next_node_label]['AT'] = next_at_label
            elif dim == 'TT':
                next_tt_label = next(self._gen_tt_label)
                cand_m_node_t[next_node_label]['TT'] = next_tt_label
            else:
                raise ValueError
        
        dis = dispersal(
            m_co_t=pd.DataFrame.from_dict(cand_m_node_t, orient='index'), 
            m_event_co=cand_m_event_node, 
            m_event_r=cand_m_event_r
        )
        imp = impurity(
            m_event_co=cand_m_event_node,
            m_event_r=cand_m_event_r
        )
        return dis, imp
    
    def _func_target(self, delta_dis, old_dis, delta_imp, old_imp):
        if old_dis == 0:
            # division-by-zero
            rr_dis = 1
        else:
            rr_dis = delta_dis / old_dis
        rr_imp = delta_imp / old_imp
        v = np.abs(rr_dis) + np.abs(rr_imp)
        return v
    
    def _pop_next_node(self):
        if len(self._leaves) > 0:
            '''
            best_node_index = None
            best_split = None
            best_node_target = -1 * np.inf
            # loop over all leaf nodes, i.e., the frontier
            for i, node in enumerate(self._leaves):
                split = self._find_attr_split(node)
                if split is None:
                    # skip a node if no split could be found for it
                    continue
                else:
                    target = split['target']
                    if target > best_node_target:
                        best_node_index = i
                        best_split = deepcopy(split)
                        best_node_target = target
            # return the leaf that leads to the largest target
            if best_node_index is not None:
                ret_node = self._leaves.pop(best_node_index)
                return ret_node, best_split
            '''
            with Pool(cpu_count() - 1) as pool:
                all_nodes = pool.map(
                    self._find_attr_split, 
                    self._leaves
                )
            all_nodes_filtered = [x for x in all_nodes if x is not None]
            if len(all_nodes_filtered) > 0:
                best_split = sorted(
                    all_nodes_filtered,
                    key=lambda x: x['target']
                )[-1]
            else:
                best_split = None
            if best_split is not None:
                best_split_node = best_split['node']
                for i, node in enumerate(self._leaves):
                    if node.label == best_split_node:
                        break
                ret_node = self._leaves.pop(i)
                return ret_node, best_split

        # return None otherwise (no more nodes or no more qualified nodes)
        return None, None
    
    def _find_attr_split(self, node):
        # locate the subset of events held by the node
        log = self._log.loc[node.event_ids]

        results_cand_attrs = []
        # loop over all candidate attributes in the pool
        for row in self.cand_attr_pool:
            result = dict()
            # copy descriptive info
            result['node'] = node.label
            for x in ['attr', 'attr_type', 'attr_dim']:
                result[x] = row[x]
            
            # generate rules for a candidate attribute
            best_rules = None
            if result['attr_type'] == 'numeric':
                # use histogram split if attribute is numeric
                best_rules = NumericRuleGenerator.HistogramSplit(
                    result['attr'], result['attr_dim'], log, bins=10
                )
            else:
                # use two subset partitioning if attribute is categorical
                # evaluate and select from over a sample of all possibilities
                # i.e., out of [2^(N-1) - 1] partitions
                best_rules = None
                best_rules_target = -1 * np.inf
                for rules in CategoricalRuleGenerator.RandomTwoSubsetPartition(
                    result['attr'], result['attr_dim'], log, 
                    n_sample=0.1, max_n_sample=100
                ):
                    dis, imp = self._evaluate_split(
                        node, rules, result['attr_dim'], log
                    )
                    delta_dis = dis - self.val_dis
                    delta_imp = imp - self.val_imp
                    target = self._func_target(
                        delta_dis, self.val_dis, delta_imp, self.val_imp
                    )
                    # find the candidate having the largest target
                    if target > best_rules_target:
                        best_rules = deepcopy(rules)
                        best_rules_target = target

            if best_rules is None:
                # no split could be found, skip this attribute
                continue
            else:
                # otherwise, compose and return
                result['split_rules'] = best_rules
            
                # evaluate
                dis, imp = self._evaluate_split(
                    node, best_rules, result['attr_dim'], log
                )
                delta_dis = dis - self.val_dis
                result['delta_dis'] = delta_dis
                delta_imp = imp - self.val_imp
                result['delta_imp'] = delta_imp
                target = self._func_target(
                    delta_dis, self.val_dis, delta_imp, self.val_imp
                )
                result['target'] = target
                results_cand_attrs.append(result)

        # select the attribute of which the split leads to highest benefit
        if len(results_cand_attrs) > 0:
            return sorted(results_cand_attrs, key=lambda x: x['target'])[-1] 
        else:
            return None
