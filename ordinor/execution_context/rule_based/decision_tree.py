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
from .Rule import Rule
from .AtomicRule import AtomicRule

class ODTMiner(BaseMiner):
    def __init__(self, el, attr_spec, max_height=-1, use_ohe=False, trace_history=False):
        self._init_miner(el, attr_spec, max_height, use_ohe, trace_history)
        self.fit_decision_tree()
        super().__init__(el)

    def _init_miner(self, el, spec, max_height, use_ohe, trace_history):
        if el is not None:
            el = check_convert_input_log(el)
        self._log = el

        # Parse specification
        if self._check_spec(spec):
            self.cand_attr_pool = spec['type_def_attrs'].copy()
            # TODO: model given rules from the specification
        
        # Filter log and keep only the relevant columns as specified
        included_cols = set({
            const.CASE_ID, const.ACTIVITY, const.TIMESTAMP, const.RESOURCE
        })
        for type_def_attr in spec['type_def_attrs'].keys():
            included_cols.add(type_def_attr)
        self._log = el[included_cols]

        # Cast data type to categorical as indicated (to boost performance)
        self._log[const.RESOURCE] = self._log[const.RESOURCE].astype('category')
        for attr, spec in spec['type_def_attrs'].items():
            if spec['attr_type'] == 'categorical':
                self._log[type_def_attr] = self._log[type_def_attr].astype('category')

        # Set max. tree height
        self.max_height = max_height

        # Set using One-Hot-Encoding to process categorical attributes
        self.use_ohe = use_ohe

        # Init dicts to track One-Hot-Encoding results (both ways)
        if self.use_ohe:
            # prefix separator for creating encoded columns
            self._ohe_prefix_sep = '_@_'
            # map original columns to derived OHE columns
            self._ohe_encoded = defaultdict(lambda: set())
            # map original columns to attribute values
            self._ohe_original_attr_vals = dict()
            # map derived OHE columns to original columns
            self._ohe_decoded = dict()
            # map derived OHE columns to original attribute values
            self._ohe_decoded_val = dict()

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
        self._height = -1

        # Init score recorders
        self.val_dis = None
        self.val_imp = None
        self.val_target = None


    def _check_spec(self, spec):
        # check if type-defining attributes are in the spec
        has_required_data = 'type_def_attrs' in spec
        if has_required_data:
            # check specification of type-defining attributes
            # NOTE: attribute names are considered unique
            for attr_name, properties in spec['type_def_attrs'].items():
                if ('attr_type' in properties and properties['attr_type'] in ['numeric', 'categorical'] and
                    'attr_dim' in properties and properties['attr_dim'] in ['CT', 'AT', 'TT']):
                    pass
                else:
                    has_required_data is False
                    break

        if not has_required_data:
            raise ValueError('invalid spec: missing required type-defining attributes or their descriptions')

        # check the partition constraint of event attributes
        is_partition_constraint_fulfilled = True
        CORR_CORE_ATTRS = {'CT': const.CASE_ID, 'AT': const.ACTIVITY, 'TT': const.TIMESTAMP}
        for attr_name, properties in spec['type_def_attrs'].items():
            corr_attr = CORR_CORE_ATTRS[properties['attr_dim']]
            # each value of the related core attr. must be mapped to exactly one candidate attr. value
            n_covered_instances = self._log.value_counts([attr_name, corr_attr], sort=False).size
            n_all_instances = self._log[corr_attr].nunique()

            is_partition_constraint_fulfilled = (
                n_covered_instances == n_all_instances
            )
            if not is_partition_constraint_fulfilled:
                raise ValueError(f"Attribute `{attr_name}` does not satisfy partition constraint on dimension `{properties['attr_dim']}`")

        return (
            is_partition_constraint_fulfilled
        )

    def _build_ctypes(self, el, **kwargs):
        self._ctypes = dict()
        for node_label, node in self._leaves.items():
            rule_ct, _, _ = node.composite_rule.to_types()
            if self.use_ohe:
                rule_ct = self._translate_ohe_rule(rule_ct)
            ct = node.ct_label
            sublog = rule_ct.apply(el)
            for case_id in sublog[const.CASE_ID].unique():
                if case_id in self._ctypes and self._ctypes[case_id] != 'CT.{}'.format(ct):
                    raise exc.InvalidParameterError(
                        param='case_attr_name',
                        reason=f'Not a case-level attribute (check case {case_id})'
                    )
                else:
                    self._ctypes[case_id] = 'CT.{}'.format(ct)
        self.is_ctypes_verified = self._verify_partition(
            set(el[const.CASE_ID].unique()), self._ctypes
        )

    def _build_atypes(self, el, **kwargs):
        self._atypes = dict()
        for node_label, node in self._leaves.items():
            _, rule_at, _ = node.composite_rule.to_types()
            if self.use_ohe:
                rule_at = self._translate_ohe_rule(rule_at)
            at = node.at_label
            sublog = rule_at.apply(el)
            for activity_label in sublog[const.ACTIVITY].unique():
                self._atypes[activity_label] = 'AT.{}'.format(at)
        self.is_atypes_verified = self._verify_partition(
            set(el[const.ACTIVITY].unique()), self._atypes
        )

    def _build_ttypes(self, el, **kwargs):
        self._ttypes = dict()
        for node_label, node in self._leaves.items():
            _, _, rule_tt = node.composite_rule.to_types()
            if self.use_ohe:
                rule_tt = self._translate_ohe_rule(rule_tt)
            tt = node.tt_label
            sublog = rule_tt.apply(el)
            for timestamp in sublog[const.TIMESTAMP].unique():
                self._ttypes[timestamp] = 'TT.{}'.format(tt)
        self.is_ttypes_verified = self._verify_partition(
            set(el[const.TIMESTAMP].unique()), self._ttypes
        )

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

        # set tree height
        self._height = 0

        # also, init scores as per root node status
        self.val_dis = 0.0
        self.val_imp = impurity(self._m_event_node, self._m_event_r)
        self.val_target = self._func_target(
            0, self.val_dis, 0, self.val_imp
        )
        return
    
    def print_scores(self):
        print('Dis. = {:.6f}'.format(self.val_dis), end=', ')
        print('Imp. = {:.6f}'.format(self.val_imp), end =', ')
        hm = 2 * self.val_dis * self.val_imp / (self.val_dis + self.val_imp)
        print('Harmonic Mean. = {:.6f}'.format(hm), end=', ')
        print('*' * 3 + f' Tree has {len(self._leaves)} leaf node(s). ' + '*' * 3)
    
    def print_tree(self):
        print('*' * 80)

        print('=' * 30 + ' TREE SUMMARY ' + '=' * 30)
        print('Score of the current tree:', end='\t')
        self.print_scores()

        print('=' * 30 + ' LEAF NODES ' + '=' * 30)
        n_total_events = sum(len(node.event_ids) for node in self._leaves.values())
        if n_total_events == len(self._log):
            print(f"{n_total_events} events were partitioned into {len(self._leaves)} leaf nodes:")
        else:
            print(f"The fitted tree is invalid: {len(self._leaves)} nodes contain {n_total_events} events ({n_total_events} != {len(self._log)})")
            return
        '''
        for node_label, node in self._leaves.items():
            print(node)
        '''
        
        print('=' * 25 + ' ENCODING TYPES WITH RULES ' + '=' * 25)
        l_rules_ct, l_rules_at, l_rules_tt = self._parse_rules_from_leaves(self._leaves)
        if self.use_ohe:
            l_rules_ct = list(map(self._translate_ohe_rule, l_rules_ct))
            l_rules_at = list(map(self._translate_ohe_rule, l_rules_at))
            l_rules_tt = list(map(self._translate_ohe_rule, l_rules_tt))
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
        # revise node data to keep unique type labels based on rules
        # NOTE: based on the rule semantics
        visited_ct = dict()
        visited_at = dict()
        visited_tt = dict()
        for label in sorted(d_leaves.keys()):
            node_ct_rule = d_leaves[label].ct_rule
            node_at_rule = d_leaves[label].at_rule
            node_tt_rule = d_leaves[label].tt_rule

            if node_ct_rule not in visited_ct:
                visited_ct[node_ct_rule] = d_leaves[label].ct_label
            else:
                d_leaves[label].ct_label = visited_ct[node_ct_rule]

            if node_at_rule not in visited_at:
                visited_at[node_at_rule] = d_leaves[label].at_label
            else:
                d_leaves[label].at_label = visited_at[node_at_rule]

            if node_tt_rule not in visited_tt:
                visited_tt[node_tt_rule] = d_leaves[label].tt_label
            else:
                d_leaves[label].tt_label = visited_tt[node_tt_rule]

        '''
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
        '''
        return d_leaves

    def _parse_rules_from_leaves(self, d_leaves):
        # parse rules from (leaf) node data
        l_rules_ct = []
        l_rules_at = []
        l_rules_tt = []
        for node_label, node in d_leaves.items():
            rule_ct, rule_at, rule_tt = node.composite_rule.to_types()
            if not rule_ct in l_rules_ct:
                l_rules_ct.append(rule_ct)
            if not rule_at in l_rules_at:
                l_rules_at.append(rule_at)
            if not rule_tt in l_rules_tt:
                l_rules_tt.append(rule_tt)
        return l_rules_ct, l_rules_at, l_rules_tt
    
    def _translate_ohe_rule(self, rule):
        # translate a rule that involves One-Hot-Encoding
        if self.use_ohe:
            rule_attrs = rule.get_attrs()
            used_enc_attrs = rule_attrs.intersection(set(self._ohe_decoded.keys()))
            visited = set()
            l_cat_ars = []
            for enc_attr in used_enc_attrs:
                if enc_attr in visited:
                    # skip if processed
                    pass
                else:
                    old_attr = self._ohe_decoded[enc_attr]
                    related_enc_attrs = self._ohe_encoded[old_attr].intersection(used_enc_attrs)
                    cat_ar_attr_vals = set()
                    old_attr_dim = None
                    for x in related_enc_attrs:
                        bool_ar = rule.subrule(x).ars[0]
                        old_attr_dim = bool_ar.attr_dim
                        # parse boolean rule meaning
                        val = {self._ohe_decoded_val[x.split(self._ohe_prefix_sep)[1]]}
                        attr_vals = (
                            val if bool_ar.attr_vals
                            else self._ohe_original_attr_vals[old_attr].difference(val)
                        )
                        if len(cat_ar_attr_vals.intersection(attr_vals)) == 0:
                            # use the union set
                            cat_ar_attr_vals.update(
                                attr_vals
                            )
                        else:
                            # use the intersection set
                            cat_ar_attr_vals = cat_ar_attr_vals.intersection(
                                attr_vals
                            )
                    cat_ar = AtomicRule(
                        attr=old_attr, attr_type='categorical',
                        attr_vals=cat_ar_attr_vals, attr_dim=old_attr_dim
                    )
                    l_cat_ars.append(cat_ar)
                    # mark as processed
                    visited.update(related_enc_attrs)
            if len(l_cat_ars) == 0:
                return Rule([AtomicRule(None)])
            else:
                return Rule(l_cat_ars)
        else:
            return rule

    def fit_decision_tree(self):
        # preprocess event log: applying One-Hot-Encoding if demanded
        if self.use_ohe is True:
            # identify categorical attributes as given in the spec
            original_cat_attrs = [attr for attr, prop in self.cand_attr_pool.items() if prop['attr_type'] == 'categorical']
            # record possible values of the categorical attributes
            for attr in original_cat_attrs:
                self._ohe_original_attr_vals[attr] = set(self._log[attr].unique())
                self._ohe_decoded_val.update({
                    (str(val), val) for val in self._log[attr].unique()
                })
            # encode data applying OHE, with original columns preserved in data
            enc_columns = pd.get_dummies(
                data=self._log[original_cat_attrs], 
                columns=original_cat_attrs,
                prefix_sep=self._ohe_prefix_sep,
                dtype=bool
            )
            self._log = pd.concat([self._log, enc_columns], axis=1)
            
            # record encoding results both ways (old->enc, enc->old)
            for old_attr in original_cat_attrs:
                for col in self._log.columns:
                    if col.startswith(f'{old_attr}{self._ohe_prefix_sep}'):
                        self._ohe_encoded[old_attr].add(col)
                        self._ohe_decoded[col] = old_attr
            print('One-Hot-Encoding applied to preprocess the following categorical attributes:')
            print('\t{}'.format(sorted(self._ohe_encoded.keys())))
            # update candidate attribute pool
            for old_attr, enc_attrs in self._ohe_encoded.items():
                for attr in enc_attrs:
                    self.cand_attr_pool[attr] = {
                        'attr_type': 'boolean',
                        'attr_dim': self.cand_attr_pool[old_attr]['attr_dim']
                    }
                del self.cand_attr_pool[old_attr]
            
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
        print(f'Start to fit decision tree with max. height = {self.max_height}')
        while True:
            if self._height == self.max_height:
                # exit search
                break

            # find the next best split
            ret = self._find_attr_split()
            if ret is None:
                print('No further split can be performed.')
                # exit search
                break
            else:
                attr, attr_type, attr_dim = ret['attr'], ret['attr_type'], ret['attr_dim']
                split_rules = ret['split_rules']
                delta_dis, delta_imp, target = (
                    ret['delta_dis'], ret['delta_imp'], ret['target']
                )
                # remove used attribute to update candidate attribute pool
                if attr_type == 'boolean':
                    del self.cand_attr_pool[attr]

            if self._decide_stopping(target):
                print('Target value ({:.6f}) meets the set criterion.'.format(
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
                    'score': self._func_solution_score(),

                    'solution': deepcopy(self._leaves)
                })

                # update tree height
                self._height += 1

                print('\t', end='')
                self.print_scores()

        # select sub-tree as output
        l_delta_score = [
            l_history[i]['score'] - l_history[i - 1]['score']
            for i, result in enumerate(l_history) if i > 0
        ]
        i_max_delta_score = np.argmax(l_delta_score)
        i_selected = i_max_delta_score + 1 + 1
        i_selected = i_selected if i_selected < len(l_history) - 1 else len(l_history) - 1
        self.val_dis = l_history[i_selected]['dispersal']
        self.val_imp = l_history[i_selected]['impurity']
        self.val_target = l_history[i_selected]['target']
        self._leaves.clear()
        self._leaves = l_history[i_selected]['solution']
        print(f'Sub-tree at step {i_selected + 1} is selected as the final solution:', end=' ')
        print('dispersal = {:.6f}, impurity = {:.6f}'.format(
            l_history[i_selected]['dispersal'], l_history[i_selected]['impurity']
        ))

        self.print_tree()

        # output history, if demanded
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
                            step + 1, 
                            result['dispersal'], result['impurity'],
                            result['target'],
                            result['score']
                        )
                    )
                    # output solution
                    fout_so.write('\n')
                    fout_so.write('=' * 79)
                    fout_so.write(f"\nStep\t[{step + 1}]\n")
                    for node_label, node in result['solution'].items():
                        fout_so.write('\n')
                        fout_so.write('-' * 79)
                        fout_so.write('\n')
                        fout_so.write(str(node))
                    fout_so.write('\n')
            print('Procedure history has been written to files.')
        
    def _func_target(self, delta_dis, old_dis, delta_imp, old_imp):
        dis = delta_dis + old_dis
        imp = delta_imp + old_imp
        # target value is expected to be maximized
        '''
        # 1. Directed Reduction Ratio (Dicing Gain)
        if old_dis == 0:
            # first split that attracts dispersal should be penalized
            rr_dis = 1.0
        else:
            rr_dis = delta_dis / old_dis
        rr_imp = delta_imp / old_imp
        v = (-1 * rr_dis) + (-1 * rr_imp)
        '''
        '''
        '''
        '''
        # 2. Negated Dispersal
        v = -1 * dis
        '''
        '''
        # 3. Absolute Change of Dispersal
        v = np.abs(delta_dis)
        '''
        # 4. Negated Harmonic Mean
        v = -1 * 2 * dis * imp / (dis + imp)

        return v
    
    def _func_solution_score(self):
        # Harmonic mean of dispersal and impurity
        return 2 * self.val_dis * self.val_imp / (self.val_dis + self.val_imp)
    
    def _decide_stopping(self, val_target):
        # run until no more split could be performed
        stop = False

        return stop
    
    def _find_attr_split(self):
        l_cand_ret = []
        # loop over all candidate attributes in the pool
        # and also the possible splits for each of them
        # a candidate to evaluate is identified by (attr, split_rules)
        for attr in self.cand_attr_pool.keys():
            result = dict()

            # copy descriptive info
            attr_type = self.cand_attr_pool[attr]['attr_type']
            attr_dim = self.cand_attr_pool[attr]['attr_dim']

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
                    elif attr_type == 'categorical':
                        # use two subset partitioning if attribute is categorical
                        # evaluate and select from over a sample of all possibilities
                        # i.e., out of [2^(n-1) - 1] partitions
                        l_cand_cat_rules = []
                        cand_rules = CategoricalRuleGenerator.RandomTwoSubsetPartition(
                            attr, attr_dim, self._log.loc[par], 
                            n_sample=100
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
                    elif attr_type == 'boolean':
                        best_rules = CategoricalRuleGenerator.BooleanPartition(
                            attr, attr_dim, self._log.loc[par], 
                        )
                    else:
                        raise ValueError

                    if len(best_rules) > 0:
                        # evalute the derived split rules by applying them
                        dis, imp = self._evaluate_split(best_rules, attr_dim)

                        # record results related to the derived rules
                        result['attr'] = attr
                        result['attr_dim'] = attr_dim
                        result['attr_type'] = attr_type
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
    
    