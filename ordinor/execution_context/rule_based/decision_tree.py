from itertools import count
from copy import deepcopy
from collections import defaultdict

import pandas as pd
import numpy as np

import ordinor.exceptions as exc
from ordinor.utils.validation import check_convert_input_log
import ordinor.constants as const

from ordinor.execution_context.base import BaseMiner

from .TreeNode import Node
from .score_funcs import dispersal, impurity
from .rule_generators import NumericRuleGenerator, CategoricalRuleGenerator

class ODTMiner(BaseMiner):
    def __init__(self, el, spec, eps, trace_history=False):
        self._init_miner(el, spec, eps, trace_history)
        self.fit_decision_tree()
        super().__init__(el)

    def _init_miner(self, el, spec, eps, trace_history):
        if el is not None:
            el = check_convert_input_log(el)
        self._log = el

        # Parse specification
        spec_checked = self._check_spec(spec)
        if spec_checked:
            self.cand_attr_pool = spec['cand_attrs']
            # TODO: model given rules from the specification
        else:
            raise ValueError('Invalid spec given')

        # Set epsilon
        self.eps = eps

        # Set tracing option
        self.trace_history = trace_history

        # Init matrices to represent states and to score:
        # Event-Resource (constant)
        self._m_event_r = el[const.RESOURCE].copy()
        # Event-Node (event - execution context)
        # (pending on init)
        self._m_event_node = pd.Series()
        # Node-Types (i.e., ct, at, tt)
        # (pending on init)
        self._m_node_t = dict()

        # Init dict for tracking splits
        self._attr_splits = defaultdict(lambda: [])

        # Init node dict
        self._root = None
        self._leaves = dict()

        # Init score recorders
        self.val_dis = None
        self.val_imp = None
        self.val_target = None
    
    def _check_spec(self, spec):
        # check if the required data is presented
        has_required_data = (
            'cand_attrs' in spec 
        )
        if not has_required_data:
            return ValueError('missing required data')

        # check the partition constraint of event attributes
        # TODO: does it have to be a func. mapping from each attr. to types?
        is_partition_constraint_fulfilled = True
        for row in spec['cand_attrs']:
            if row['attr_dim'] == 'CT':
                corr_attr = const.CASE_ID
            elif row['attr_dim'] == 'AT':
                corr_attr = const.ACTIVITY
            elif row['attr_dim'] == 'TT':
                corr_attr = const.TIMESTAMP
            else:
                raise ValueError(f"`{row}` has invalid corresponding type")

            visited_corr_attr_vals = list()
            for grouped, _ in self._log.groupby([row['attr'], corr_attr]):
                u, v = grouped[0], grouped[1]
                visited_corr_attr_vals.append(v)
            
            is_partition_constraint_fulfilled = (
                len(visited_corr_attr_vals) == len(pd.unique(self._log[corr_attr]))
            )
            if not is_partition_constraint_fulfilled:
                raise ValueError(f"Attribute `{row['attr']}` does not satisfy partition constraint")

        return (
            is_partition_constraint_fulfilled
        )

    def _build_ctypes(self, el, **kwargs):
        el = check_convert_input_log(el)
        self._ctypes = dict()
        for node_label, node in self._leaves.items():
            rule_ct, _, _ = node.composite_rule.to_types()
            ct = node.ct_label
            sublog = rule_ct.apply(el)
            for case_id in set(sublog[const.CASE_ID]):
                if case_id in self._ctypes and self._ctypes[case_id] != 'CT.{}'.format(ct):
                    raise exc.InvalidParameterError(
                        param='case_attr_name',
                        reason=f'Not a case-level attribute (check case {case_id})'
                    )
                else:
                    self._ctypes[case_id] = 'CT.{}'.format(ct)
        self.is_ctypes_verified = self._verify_partition(
            set(el[const.CASE_ID]), self._ctypes)

    def _build_atypes(self, el, **kwargs):
        el = check_convert_input_log(el)
        self._atypes = dict()
        for node_label, node in self._leaves.items():
            _, rule_at, _ = node.composite_rule.to_types()
            at = node.at_label
            sublog = rule_at.apply(el)
            for activity_label in set(sublog[const.ACTIVITY]):
                self._atypes[activity_label] = 'AT.{}'.format(at)
        self.is_atypes_verified = self._verify_partition(
            set(el[const.ACTIVITY]), self._atypes)

    def _build_ttypes(self, el, **kwargs):
        el = check_convert_input_log(el)
        self._ttypes = dict()
        for node_label, node in self._leaves.items():
            _, _, rule_tt = node.composite_rule.to_types()
            tt = node.tt_label
            sublog = rule_tt.apply(el)
            for timestamp in set(sublog[const.TIMESTAMP]):
                self._ttypes[timestamp] = 'TT.{}'.format(tt)
        self.is_ttypes_verified = self._verify_partition(
            set(el[const.TIMESTAMP]), self._ttypes)

    def _init_label_generators(self):
        # Init label generators for nodes creation (
        # label starts from "1" (different from the root "0")
        self._gen_node_label = count(start=1, step=1)
        self._gen_ct_label = count(start=1, step=1)
        self._gen_at_label = count(start=1, step=1)
        self._gen_tt_label = count(start=1, step=1)
    
    def _init_params(self):
        # Event-Node: all events belong to a single node (no split exists)
        self._m_event_node = pd.Series(
            [0] * len(self._m_event_r), 
            index=self._m_event_r.index
        )
        # Node-Types: no types exist
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
            parent_label=None,
            step_rule=None, parent_rule=None
        )
        # add root to frontier
        self._leaves[0] = root
        # update Node-Types
        self._m_node_t[root.label] = {
            'CT': root.ct_label, 
            'AT': root.at_label, 
            'TT': root.tt_label
        }
        # set tree root
        self._root = root

        # also, init scores as per root node status
        self.val_dis = 0.0
        self.val_imp = impurity(self._m_event_node, self._m_event_r)
        self.val_target = self._func_target(
            0, self.val_dis, 0, self.val_imp
        )
        return
    
    def print_scores(self):
        print('Dis. = {:.6f}'.format(self.val_dis), end=', ')
        print('Imp. = {:.6f}'.format(self.val_imp), end=', ')
    
    def print_tree(self):
        print('*' * 80)
        print('=' * 30 + ' TREE SUMMARY ' + '=' * 30)

        '''
        # TODO
        l_all_nodes = list(self.traverse_tree())
        print(f"Number of nodes:\t{len(l_all_nodes)}")
        '''

        print('Current scores:', end='\t')
        self.print_scores()

        print('=' * 30 + ' LEAF NODES ' + '=' * 30)
        n_total_events = sum(len(node.event_ids) for node in self._leaves.values())
        if n_total_events == len(self._log):
            print(f"{n_total_events} events were partitioned into {len(self._leaves)} leaf nodes:")
        else:
            print(f"The fitted tree is invalid: {len(self._leaves)} nodes contain {n_total_events} events ({n_total_events} != {len(self._log)})")
            return
        for node_label, node in self._leaves.items():
            print(node)
        
        print('=' * 25 + ' ENCODING TYPES WITH RULES ' + '=' * 25)
        l_rules_ct = []
        l_rules_at = []
        l_rules_tt = []
        for node_label, node in self._leaves.items():
            rule_ct, rule_at, rule_tt = node.composite_rule.to_types()
            has_eq_rule_ct = False
            for rule in l_rules_ct:
                if rule == rule_ct:
                    has_eq_rule_ct = True
                    break
            has_eq_rule_at = False
            for rule in l_rules_at:
                if rule == rule_at:
                    has_eq_rule_at = True
                    break
            has_eq_rule_tt = False
            for rule in l_rules_tt:
                if rule == rule_tt:
                    has_eq_rule_tt = True
                    break
            if not has_eq_rule_ct:
                l_rules_ct.append(rule_ct)
            if not has_eq_rule_at:
                l_rules_at.append(rule_at)
            if not has_eq_rule_tt:
                l_rules_tt.append(rule_tt)
        print('Rules for Case Types:')
        for r in l_rules_ct:
            print(f"\t{r}")
        print('Rules for Activity Types:')
        for r in l_rules_at:
            print(f"\t{r}")
        print('Rules for Time Types:')
        for r in l_rules_tt:
            print(f"\t{r}")

        print('*' * 80)

    def _revise_leaf_labels(self, d_leaves):
        # revise node data and keep unique type labels based on rules
        node_labels = sorted(d_leaves.keys())
        for i in range(len(node_labels) - 1):
            ref_label = node_labels[i]
            ref_node = d_leaves[ref_label]
            for j in range(i + 1, len(node_labels)):
                cmp_label = node_labels[j]
                cmp_node = d_leaves[cmp_label]
                if cmp_node.ct_rule == ref_node.ct_rule:
                    cmp_node.ct_label = ref_node.ct_label
                if cmp_node.at_rule == ref_node.at_rule:
                    cmp_node.at_label = ref_node.at_label
                if cmp_node.tt_rule == ref_node.tt_rule:
                    cmp_node.tt_label = ref_node.tt_label

        return d_leaves

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
        self._init_params()
        self._init_label_generators()
        self._init_leaves()
        print('Decision tree initialized with an empty root node\n', end='\t')
        self.print_scores()

        # set recorders for tracing
        l_history = []

        # iterative tree induction
        print(f'Start to fit decision tree with epsilon = {self.eps}')
        while True:
            # find the next best split
            ret = self._find_attr_split()
            if ret is None:
                print('No further split can be performed.')
                # exit search
                break
            else:
                attr = ret['attr']
                attr_dim = ret['attr_dim']
                split_rules = ret['split_rules']
                delta_dis, delta_imp, target = (
                    ret['delta_dis'], ret['delta_imp'], ret['target']
                )

            if self._decide_stopping(target):
                print('The target value ({:.6f}) is too insignificant.'.format(
                    target
                ))
                # exit search
                break
            else:
                print(f"Tree grows by splitting all current leaf nodes on `{attr}`")

                # clear existing split tracker
                # this needs to be reconstructed based on the new leaves
                self._attr_splits[attr].clear()

                # apply the split (rules) found and create child nodes
                l_new_leaf_nodes = []
                id_curr_leaf_nodes = sorted(self._leaves.keys())
                for node_label in id_curr_leaf_nodes:
                    node = self._leaves[node_label]
                    # locate subset of log to be split
                    sublog = self._log.loc[node.event_ids]
                    # set flag to identify if node is affected
                    is_node_split = False
                    for rule in split_rules:
                        # apply split on the node
                        par = rule.apply(sublog, index_only=True)

                        if len(par) == 0:
                            # skip if current split is not related
                            continue
                        else:
                            is_node_split = True
                            # create a child node per each split
                            child_node_label = next(self._gen_node_label)
                            # assign event subset to the child node
                            # by default type labels are inherited from the parent 
                            child_node = Node(
                                label=child_node_label,
                                event_ids=par,
                                ct_label=node.ct_label, at_label=node.at_label, tt_label=node.tt_label,
                                parent_label=node_label,
                                step_rule=rule,
                                parent_rule=node.composite_rule
                            )
                            # update type labels depending on the rule applied
                            # overwrite type labels based on attr dim
                            if attr_dim == 'CT':
                                new_ct_label = next(self._gen_ct_label)
                                child_node.ct_label = new_ct_label
                            elif attr_dim == 'AT':
                                new_at_label = next(self._gen_at_label)
                                child_node.at_label = new_at_label
                            elif attr_dim == 'TT':
                                new_tt_label = next(self._gen_tt_label)
                                child_node.tt_label = new_tt_label
                            else:
                                raise ValueError
                            # TODO: attach to the tree
                            #node.append_child(child_node)
                            # update Event-Node
                            self._m_event_node.loc[par] = child_node_label
                            # record new leaf
                            l_new_leaf_nodes.append(child_node)

                    # return the node to leaves if none of the splits applied
                    if is_node_split is False:
                        l_new_leaf_nodes.append(node)

                # update leaves: remove previous nodes and add new ones
                self._leaves.clear()
                for node in l_new_leaf_nodes:
                    self._leaves[node.label] = node
                
                # revise node data
                self._leaves = self._revise_leaf_labels(self._leaves)

                self._m_node_t.clear()
                for node_label, node in self._leaves.items():
                    # update matrix: Node-Types
                    self._m_node_t[node_label] = {
                        'CT': node.ct_label, 
                        'AT': node.at_label, 
                        'TT': node.tt_label
                    }
                    # update existing split tracker                
                    subrule = node.composite_rule.subrule(attr)
                    if subrule not in self._attr_splits[attr]:
                        self._attr_splits[attr].append(subrule)

                # update scores
                self.val_dis += delta_dis
                self.val_imp += delta_imp
                self.val_target = target

                # record step result
                l_history.append({
                    'dispersal': self.val_dis,
                    'impurity': self.val_imp,
                    'target': self.val_target,

                    'solution': deepcopy(self._leaves)
                })

                print('\t', end='')
                self.print_scores()

        # print tree
        print('Procedure stopped with final scores:')
        self.print_tree()
        if self.trace_history:
            ts_now = pd.Timestamp.now()
            fname_prefix = 'ODTMiner_{}_'.format(
                ts_now.strftime('%Y%m%d-%H%M%S')
            )
            # print history (stats and solutions), indexed by step number
            fname_stats = fname_prefix + 'stats.out'
            fname_sol = fname_prefix + 'solutions.out'
            with open(fname_stats, 'w') as fout_st, open(fname_sol, 'w') as fout_so:
                fout_st.write('step,dispersal,impurity,target\n')
                for step, result in enumerate(l_history):
                    # output stats
                    fout_st.write(
                        '{},{:.6f},{:.6f},{:.6f}\n'.format(
                            step, 
                            result['dispersal'], result['impurity'],
                            result['target']
                        )
                    )
                    # output solution
                    fout_so.write('\n')
                    fout_so.write('=' * 79)
                    fout_so.write(f"\nStep\t[{step}]\n")
                    for node_label, node in result['solution'].items():
                        fout_so.write('\n')
                        fout_so.write('-' * 79)
                        fout_so.write('\n')
                        fout_so.write(str(node))
                    fout_so.write('\n')

    def _func_target(self, delta_dis, old_dis, delta_imp, old_imp):
        # target value is expected to be maximized
        '''
        # Undirected Reduction Ratio
        if old_dis == 0:
            # division-by-zero
            rr_dis = 1
        else:
            rr_dis = delta_dis / old_dis
        rr_imp = delta_imp / old_imp
        v = np.abs(rr_dis) + np.abs(rr_imp)
        '''
        '''
        # Directed Reduction Ratio
        if old_dis == 0:
            # division-by-zero
            rr_dis = 1
        else:
            rr_dis = delta_dis / old_dis
        rr_imp = delta_imp / old_imp
        v = -1 * (rr_dis + rr_imp)
        '''
        '''
        # Dispersal only
        v = -1 * (delta_dis + old_dis)
        '''
        '''
        # Change of Dispersal
        v = np.abs(delta_dis)
        '''
        # Harmonic Mean
        dis = delta_dis + old_dis
        imp = delta_imp + old_imp
        v = -1 * 2 * dis * imp / (dis + imp)

        return v
    
    def _func_score(self, dis, imp):
        # harmonic mean (with conditions on the extremes)
        if dis == 0 or imp == 0.0:
            return 1.0
        return 2 * dis * imp / (dis + imp)

    def _decide_stopping(self, val_target):
        #return val_target < self.eps
        # run until no more split could be performed
        return False
    
    def _find_attr_split(self):
        l_cand_ret = []
        # loop over all candidate attributes in the pool
        # and also the possible splits for each of them
        # a candidate to evaluate is identified by (attr, split_rules)
        for row in self.cand_attr_pool:
            result = dict()

            # copy descriptive info
            attr = row['attr']
            attr_type = row['attr_type']
            attr_dim = row['attr_dim']

            l_idx_sublog = []
            # split should be found from existing partitions on
            # each of the cand. attribute, NOT directly from the leaves
            existing_split_rules = self._attr_splits[attr]
            if len(existing_split_rules) == 0:
                # use the entire log if no partition exists (of this attr)
                l_idx_sublog.append(self._log.index) 
            else:
                # for each cell in the existing partition (of this attr)
                for existing_rule in existing_split_rules:
                    # identify the subset of the log
                    l_idx_sublog.append(
                        existing_rule.apply(self._log, index_only=True)
                    )

            for par in l_idx_sublog:
                if len(par) > 0:
                    # determine possible further splits
                    if attr_type == 'numeric':
                        # use histogram for numeric attribute
                        best_rules = NumericRuleGenerator.HistogramSplit(
                            attr, attr_dim, self._log.loc[par], bins='fd'
                        )
                    else:
                        # use two subset partitioning if attribute is categorical
                        # evaluate and select from over a sample of all possibilities
                        # i.e., out of [2^(N-1) - 1] partitions
                        l_cand_cat_rules = []
                        cand_rules = CategoricalRuleGenerator.RandomTwoSubsetPartition(
                            attr, attr_dim, self._log.loc[par], 
                            n_sample=None, max_n_sample=2048
                        )
                        for rules in cand_rules:
                            dis, imp = self._evaluate_split(rules, attr_dim)
                            delta_dis = dis - self.val_dis
                            delta_imp = imp - self.val_imp
                            target = self._func_target(
                                delta_dis, self.val_dis, delta_imp, self.val_imp
                            )
                            l_cand_cat_rules.append((rules, target))
                        
                        if len(l_cand_cat_rules) > 0:
                            best_rules = max(l_cand_cat_rules, key=lambda x: x[1])[0]
                        else:
                            best_rules = []

                    if len(best_rules) > 0:
                        # evalute the derived split rules by applying them
                        dis, imp = self._evaluate_split(best_rules, attr_dim)

                        # record results related to the derived rules
                        result['attr'] = attr
                        result['attr_dim'] = attr_dim
                        result['split_rules'] = best_rules
                        # 
                        delta_dis = dis - self.val_dis
                        result['delta_dis'] = delta_dis
                        delta_imp = imp - self.val_imp
                        result['delta_imp'] = delta_imp
                        target = self._func_target(
                            delta_dis, self.val_dis, delta_imp, self.val_imp
                        )
                        result['target'] = target
                        l_cand_ret.append(result)

        # select the attribute of which the split leads to highest benefit
        if len(l_cand_ret) > 0:
            ret = max(l_cand_ret, key=lambda x: x['target'])
            return ret
        else:
            return None

    def _evaluate_split(self, split_rules, attr_dim, nodes=None):
        # everything within the scope of this function operates on copies

        # get deep copies for evaluation
        cand_m_event_r, cand_m_event_node, cand_m_node_t = self._get_matrices_copy()

        # all leave nodes are used if no nodes are specified
        nodes = deepcopy(self._leaves) if nodes is None else nodes

        # create "virtual" child nodes
        cand_leaves = dict()
        for node_label, node in nodes.items():
            # apply to each node
            is_node_split = False
            for rule in split_rules:
                # locate the subset and apply the rule
                node_log = self._log.loc[node.event_ids]
                par = rule.apply(node_log, index_only=True)

                if len(par) == 0:
                    # skip if current split is not related
                    continue
                else:
                    is_node_split = True
                    # create "virtual" child node
                    child_node_label = next(self._gen_node_label)
                    cand_m_event_node.loc[par] = child_node_label

                    child_node = Node(
                        label=child_node_label,
                        event_ids=par,
                        ct_label=node.ct_label, at_label=node.at_label, tt_label=node.tt_label,
                        parent_label=node.label,
                        step_rule=rule,
                        parent_rule=node.composite_rule
                    )

                    if attr_dim == 'CT':
                        new_ct_label = next(self._gen_ct_label)
                        child_node.ct_label = new_ct_label
                    elif attr_dim == 'AT':
                        new_at_label = next(self._gen_at_label)
                        child_node.at_label = new_at_label
                    elif attr_dim == 'TT':
                        new_tt_label = next(self._gen_tt_label)
                        child_node.tt_label = new_tt_label
                    else:
                        raise ValueError
                    # add the virtual child to the queue
                    # "peek" update the frontiers
                    cand_leaves[child_node_label] = child_node

            if is_node_split is False:
                cand_leaves[node.label] = deepcopy(node)

        # revise node data
        cand_leaves = self._revise_leaf_labels(cand_leaves)
        # update matrix: Node-Types
        cand_m_node_t.clear()
        for node_label, node in cand_leaves.items():
            cand_m_node_t[node_label] = {
                'CT': node.ct_label, 
                'AT': node.at_label, 
                'TT': node.tt_label
            }
        
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
    
    