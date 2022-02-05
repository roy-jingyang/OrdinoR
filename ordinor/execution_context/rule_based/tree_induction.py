from itertools import count
from copy import deepcopy

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
    def __init__(self, el, specification):
        # Read event log
        if el is not None:
            el = check_convert_input_log(el)
        self._log = el

        # Parse specification
        self.cand_attr_pool = specification['cand_attrs']
        # TODO: model given rules from the specification

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
        self._node_list = list()

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

    def _init_node_list(self):
        # clear list
        self._node_list.clear()

        # build root node (all labels marked "0")
        # root node contains all events
        # root node is marked with an empty atomic rule
        null_ar = AtomicRule()
        root = Node(
            label=0, event_ids=self._log.index, parent_node=None,
            atomic_rule=null_ar,
            ct_label=0, at_label=0, tt_label=0
        )
        self._node_list.append(root)
        self._m_node_t[root.label] = {
            'CT': root.ct, 'AT': root.at, 'TT': root.tt
        }
        self.root = root

        # also, init scores as per root node status
        self.val_dis = 0.0
        self.val_imp = impurity(self._m_event_node, self._m_event_r)
        self.val_target = -1 * np.inf

        return
    
    def _report_scores(self):
        print('Dis. = {:.6f}'.format(self.val_dis), end=', ')
        print('Imp. = {:.6f}'.format(self.val_imp), end=', ')
        print('Target ={:.6f}'.format(self.val_target))
    
    def _fit_decision_tree(self, el, eps=0.1):
        # main procedure
        # initialize
        self._init_matrices()
        self._init_label_generators()
        self._init_node_list()
        print('Decision tree initialized with empty root node\n', end='\t')
        self._report_scores()

        # iterative tree induction
        print(f'Start to fit decision tree with epsilon = {eps}')
        tree_expandable = True
        while tree_expandable:
            next_node, split = self._find_next_node(self._node_list)

            if next_node is None:
                print('There is no more next node to be split')
                tree_expandable = False
                print('Tree fitted. Procedure stops with final scores:\n', end='\t')
            else:
                print(f"Tree grows by splitting node ({next_node.label}) on attribute {split['attr']}")
                log = self.log.loc[next_node.event_ids]

                # apply the split (rules) found
                for rule in split['rules']:
                    child_node_label = next(self._gen_node_label)
                    par = rule.apply(log, index_only=True)
                    # set (new) type labels depending on the rule applied
                    # by default type labels are inherited from parent 
                    new_ct_label = next_node.ct
                    new_at_label = next_node.at
                    new_tt_label = next_node.tt
                    if split['attr_dim'] == 'CT':
                        new_ct_label = next(self._gen_ct_label)
                        self._m_node_t[child_node_label]['CT'] = new_ct_label
                    elif split['attr_dim'] == 'AT':
                        new_at_label = next(self._gen_at_label)
                        self._m_node_t[child_node_label]['AT'] = new_at_label
                    elif split['attr_dim'] == 'TT':
                        new_tt_label = next(self._gen_tt_label)
                        self._m_node_t[child_node_label]['TT'] = new_tt_label
                    else:
                        raise ValueError
                    
                    # create child node
                    child_node = Node(
                        label=child_node_label,
                        event_ids=par, atomic_rule=rule,
                        ct_label=new_ct_label, at_label=new_at_label, tt_label=new_tt_label
                    )
                    # attach to the tree
                    next_node.append_child(child_node)
                    # add to queue
                    self._node_list.append(child_node)

                # remove next node used from next interation
                del self._m_node_t[next_node.label]

                # update scores
                self.val_dis += split['delta_dis']
                self.val_imp += split['delta_imp']
                self.val_target = split['target']

            self._report_scores()

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
            cand_m_node_t[next_node_label] = {'CT': node.ct, 'AT': node.at, 'TT': node.tt}
            
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
        v = -1 * (rr_dis + rr_imp)
        return v
    
    def _find_next_node(self):
        # TODO
        return next_node, split
    
    def _find_attr_split(self, node):
        log = self._log.loc[node.event_ids]
        results_cand_attrs = []

        for row in self.cand_attr_pool:
            result = dict()
            # copy descriptive info
            for x in ['attr', 'attr_type', 'attr_dim']:
                result[x] = row[x]
            
            # select rules from candidate splits
            best_rules = None
            if result['attr_type'] == 'numeric':
                best_rules = NumericRuleGenerator.HistogramSplit(
                    result['attr'], result['attr_dim'], log, bins=10
                )
            else:
                # evaluate over a sample out of all possibilities
                # x% of [2^(N-1) - 1] partitions
                curr_target = -np.inf
                curr_rules = None
                for rules in CategoricalRuleGenerator.RandomTwoSubsetPartition(
                    result['attr'], result['attr_dim'], log, n_sample=10
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
                    if target > curr_target:
                        curr_rules = rules
                        curr_target = target
                if curr_target > self.val_target:
                    best_rules = curr_rules

        return result

