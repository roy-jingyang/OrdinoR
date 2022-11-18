'''
A series of search-based solutions:
    - greedy descent
    - tabu search
    - simulated annealing
'''

from collections import defaultdict, deque
from itertools import product, combinations, islice
from math import comb as num_combinations

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import pdist, squareform

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

from ordinor.execution_context.base import BaseMiner

from .TreeNode import Node2
from .AtomicRule import AtomicRule
from .Rule import Rule

class BaseSearchMiner(BaseMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random',
        init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
    ):
        # Parse log
        if el is not None:
            self._log = check_convert_input_log(el).copy()
            # only use events with resource information
            self._log = self._log[self._log[const.RESOURCE].notna()]
            print('Use only {} events with resource information recorded.'.format(len(self._log)))
        else:
            raise ValueError('invalid log')

        # Parse attribute specification
        if self._check_spec(attr_spec):
            # Record type-defining attributes and values at fixed ordering
            # NOTE: lists of type-defining attributes for C/A/T dimensions
            self._tda_case = []
            self._tda_act = []
            self._tda_time = []
            included_cols = [
                const.CASE_ID, const.ACTIVITY, const.TIMESTAMP, const.RESOURCE
            ]
            # NOTE: set of dimensions included in the specification
            for attr, spec in attr_spec['type_def_attrs'].items():
                if attr in included_cols:
                    included_cols.remove(attr)
                if spec['attr_type'] == 'categorical':
                    if spec['attr_dim'] == 'CT':
                        self._tda_case.append(attr)
                    elif spec['attr_dim'] == 'AT':
                        self._tda_act.append(attr)
                    elif spec['attr_dim'] == 'TT':
                        self._tda_time.append(attr)
                    else:
                        raise ValueError(f'Unrecognized attribute dimension `{spec["attr_dim"]}`')
            # sort to preserve fixed ordering
            self._tda_case.sort()
            self._tda_act.sort()
            self._tda_time.sort()
            # filter log and keep only the relevant columns as specified
            self._log = self._log[
                included_cols +
                self._tda_case + self._tda_act + self._tda_time
            ]
            self._n_tda_case = len(self._tda_case)
            self._n_tda_act = len(self._tda_act)
            self._n_tda_time = len(self._tda_time)
            self._n_tda = self._n_tda_case + self._n_tda_act + self._n_tda_time

            # NOTE: `tdav', a map for type-defining attribute names and values
            # NOTE: `tda_dim`, a reverse map for `tda_*` 
            # NOTE: `index_tda_*`, maps for the relative indices of attribute values 
            # NOTE: `par`, a map for the partition over attribute values,
            #              represented as squared matrices, initialized later
            self._tdav = dict()
            self._tda_dim = dict()
            self._index_tda_case = dict()
            self._index_tda_act = dict()
            self._index_tda_time = dict()
            self._pars = dict()
            for dim, tda in enumerate([self._tda_case, self._tda_act, self._tda_time]):
                start = 0
                for attr in tda:
                    attr_values = sorted(self._log[attr].unique())
                    self._tdav[attr] = attr_values
                    n_attr_values = len(attr_values)
                    if dim == 0:
                        self._tda_dim[attr] = 'CT'
                        self._index_tda_case[attr] = tuple((start, start+n_attr_values))
                    elif dim == 1:
                        self._tda_dim[attr] = 'AT'
                        self._index_tda_act[attr] = tuple((start, start+n_attr_values))
                    else:
                        self._tda_dim[attr] = 'TT'
                        self._index_tda_time[attr] = tuple((start, start+n_attr_values))
                    start += n_attr_values
            # NOTE: `index_*`, maps for the absolute indices for dimensions
            self._index_case = tuple((0, self._index_tda_case[self._tda_case[-1]][1]))
            self._index_act = tuple((self._index_case[1], self._index_case[1]+self._index_tda_act[self._tda_act[-1]][1]))
            self._index_time = tuple((self._index_act[1], self._index_act[1]+self._index_tda_time[self._tda_time[-1]][1]))

            # Record log-related constants
            # NOTE: |E| == |E_res|, log size (after excluding events with no resource)
            self._n_E = len(self._log)
            # NOTE: absolute frequencies of resources |E_r|
            self._p_res = self._log[const.RESOURCE].value_counts(
                normalize=False, sort=False
            ).to_dict()
            # NOTE: integers, to be used for calculating dispersal
            self._ncomb_E_res_2 = dict(
                (r, num_combinations(count_r, 2)) for r, count_r in self._p_res.items()
            )
            # NOTE: an |E| x 1 array recording originator resources
            self._arr_event_r = self._log[const.RESOURCE].to_numpy(copy=True)

            # normalize the frequencies of resources
            for r in self._p_res.keys():
                self._p_res[r] /= self._n_E

            # NOTE: float constant, to be used for calculating impurity
            self._max_impurity = scipy_entropy(
                np.array(list(self._p_res.values()), dtype=np.double), base=2
            )

            # NOTE: diameter of the search space,
            #       shortest distance between the farthest states 
            #       ('zero' and 'full_split'), by #moves
            self._search_diameter = sum([len(tdav) - 1 for tdav in self._tdav])

            # drop non-type-defining attributes columns
            self._log.drop(columns=included_cols, inplace=True)
            # apply One-Hot-Encoding to the log (pandas.get_dummies)
            # NOTE: use consistent prefix separator
            self._ohe_prefix_sep = '^^@^^'
            self._log = pd.get_dummies(
                self._log, columns=self._log.columns, 
                prefix_sep=self._ohe_prefix_sep,
                sparse=True
            )

            # verify if encoded columns conform to the saved ordering
            saved_attr_values = []
            for attr in (self._tda_case + self._tda_act + self._tda_time):
                for attr_val in self._tdav[attr]:
                    saved_attr_values.append(f'{attr}{self._ohe_prefix_sep}{attr_val}')
            if saved_attr_values == list(self._log.columns):
                pass
            else:
                return ValueError('log columns after encoding does not conform to the saved ordering')

            # convert the log to a numpy array
            self._log = self._log.to_numpy(dtype=bool)

            # set random number generator
            self._rng = (
                np.random.default_rng() if random_number_generator is None 
                else random_number_generator
            )

            # Record search status
            # current nodes (execution contexts)
            self._nodes = []

            # Record final results
            self.type_dict = dict()

            # Set init method
            self.init_method = init_method
            self.init_batch = init_batch

            # Set flags
            # whether to include sparsity in search
            #self.use_sparsity = use_sparsity
            # whether to print the search procedure
            self.print_steps = print_steps
            # whether to record the history of intermediate stats
            self.trace_history = trace_history

            # TODO: model user-supplied categorization rules
            self._init_state(init_method=self.init_method)
            self._search()

            super().__init__(el)

    def _check_spec(self, spec):
        # check if type-defining attributes are in the spec
        has_required_data = 'type_def_attrs' in spec
        if has_required_data:
            # check specification of type-defining attributes
            # NOTE: attribute names are considered unique
            # TODO: support numeric attributes
            for attr_name, properties in spec['type_def_attrs'].items():
                if ('attr_type' in properties and properties['attr_type'] in ['categorical'] and
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
        ctype_arr_tda = set()
        for node in self._nodes:
            arr_case = node.arr[slice(*self._index_case)]
            ctype_arr_tda.add(tuple(arr_case))
        self._ctypes = dict()
        #print(ctype_arr_tda)
        for i, k_arr_case in enumerate(ctype_arr_tda):
            arr_case = np.array(k_arr_case, dtype=np.bool)
            ct_label = i
            # translate to rules and save
            ct_rules = []
            for attr in self._tda_case:
                arr_case_attr = arr_case[slice(*self._index_tda_case[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_case_attr)[0])
                if len(sel_attr_values) < len(self._tdav[attr]):
                    # only save rules if they evaluate to partitions
                    ct_rules.append(
                        AtomicRule(attr=attr, attr_type='categorical', attr_vals=sel_attr_values, attr_dim='CT')
                    )
            ct_rules = Rule(ars=ct_rules)

            sel_events = self._apply_to_part(
                arr=arr_case, n_attrs=self._n_tda_case, 
                rows=None, cols=slice(*self._index_case)
            )
            for case_id in el[const.CASE_ID].iloc[sel_events].unique():
                if case_id in self._ctypes and self._ctypes[case_id] != 'CT.{}'.format(ct_label):
                    raise exc.InvalidParameterError(
                        param='case_attr_name',
                        reason=f'Not a case-level attribute (check case {case_id})'
                    )
                else:
                    self._ctypes[case_id] = 'CT.{}'.format(ct_label)
            self.type_dict['CT.{}'.format(ct_label)] = ct_rules
        #print(self.type_dict)
        self.is_ctypes_verified = self._verify_partition(
            set(el[const.CASE_ID].unique()), self._ctypes
        )

    def _build_atypes(self, el, **kwargs):
        atype_arr_tda = set()
        for node in self._nodes:
            arr_act = node.arr[slice(*self._index_act)]
            atype_arr_tda.add(tuple(arr_act))
        self._atypes = dict()
        #print(atype_arr_tda)
        for i, k_arr_act in enumerate(atype_arr_tda):
            arr_act = np.array(k_arr_act, dtype=np.bool)
            at_label = i
            # translate to rules and save
            at_rules = []
            for attr in self._tda_act:
                arr_act_attr = arr_act[slice(*self._index_tda_act[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_act_attr)[0])
                if len(sel_attr_values) < len(self._tdav[attr]):
                    # only save rules if they evaluate to partitions
                    at_rules.append(
                        AtomicRule(attr=attr, attr_type='categorical', attr_vals=sel_attr_values, attr_dim='AT')
                    )
            at_rules = Rule(ars=at_rules)

            sel_events = self._apply_to_part(
                arr=arr_act, n_attrs=self._n_tda_act, 
                rows=None, cols=slice(*self._index_act)
            )
            for activity_label in el[const.ACTIVITY].iloc[sel_events].unique():
                self._atypes[activity_label] = 'AT.{}'.format(at_label)
            self.type_dict['AT.{}'.format(at_label)] = at_rules
        #print(self.type_dict)
        self.is_atypes_verified = self._verify_partition(
            set(el[const.ACTIVITY].unique()), self._atypes
        )

    def _build_ttypes(self, el, **kwargs):
        ttype_arr_tda = set()
        for node in self._nodes:
            arr_time = node.arr[slice(*self._index_time)]
            ttype_arr_tda.add(tuple(arr_time))
        self._ttypes = dict()
        #print(ttype_arr_tda)
        for i, k_arr_time in enumerate(ttype_arr_tda):
            arr_time = np.array(k_arr_time, dtype=np.bool)
            tt_label = i
            # translate to rules and save
            tt_rules = []
            for attr in self._tda_time:
                arr_time_attr = arr_time[slice(*self._index_tda_time[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_time_attr)[0])
                if len(sel_attr_values) < len(self._tdav[attr]):
                    # only save rules if they evaluate to partitions
                    tt_rules.append(
                        AtomicRule(attr=attr, attr_type='categorical', attr_vals=sel_attr_values, attr_dim='TT')
                    )
            tt_rules = Rule(ars=tt_rules)

            sel_events = self._apply_to_part(
                arr=arr_time, n_attrs=self._n_tda_time, 
                rows=None, cols=slice(*self._index_time)
            )
            for timestamp in el[const.TIMESTAMP].iloc[sel_events].unique():
                self._ttypes[timestamp] = 'TT.{}'.format(tt_label)
            self.type_dict['TT.{}'.format(tt_label)] = tt_rules
        #print(self.type_dict)
        self.is_ttypes_verified = self._verify_partition(
            set(el[const.TIMESTAMP].unique()), self._ttypes
        )

    def _init_state(self, **kwargs):
        # initialize partitions on all attributes
        # initialize nodes correspondingly
        comb = []
        total_comb_size = 1
        for i, tda in enumerate([self._tda_case, self._tda_act, self._tda_time]):
            tda_i = []
            comb_i = []
            if len(tda) == 0:
                continue
            for attr in tda:
                attr_parts = []
                n = len(self._tdav[attr])
                if n == 1:
                    self._pars[attr] = np.array([[1]], dtype=bool)
                    attr_parts.append(np.array([1], dtype=bool))
                else:
                    if self.init_method == 'zero':
                        self._pars[attr] = np.zeros((n, n), dtype=bool)
                        # apply no partition to all attributes (one-holds-all)
                        self._pars[attr][:,0] = 1
                        attr_parts.append(self._pars[attr][:,0])
                    elif self.init_method == 'full_split':
                        # create a singleton for each value
                        self._pars[attr] = np.eye(n, dtype=bool)
                        for j in range(n):
                            attr_parts.append(self._pars[attr][:,j])
                    elif self.init_method == 'random':
                        self._pars[attr] = np.zeros((n, n), dtype=bool)
                        # select a part for each attribute value randomly
                        for i in range(n):
                            i_part = self._rng.choice(n)
                            self._pars[attr][i,i_part] = 1
                        for j in range(n):
                            if self._pars[attr][:,j].any():
                                attr_parts.append(self._pars[attr][:,j])
                    else:
                        # TODO: model user-supplied categorization rules
                        # init_method == 'informed'
                        raise NotImplementedError
                tda_i.append(attr_parts)
            for prod in product(*tda_i):
                comb_i.append(np.concatenate(prod))
            comb.append(comb_i)
            total_comb_size *= len(comb_i)
        
        if self.init_method == 'full_split':
            # TODO: need to start from existing combinations instead of enumeration
            raise NotImplementedError('Potential oversized problem: to develop a mechanism to include only observed combinations')
        else:
            '''
            # Solution 1.1 (fallback): enumerate and test all combinations
            for prod in product(*comb):
                arr_joined = np.concatenate(prod)
                events = self._apply_to_all(arr_joined)
                if len(events) > 0:
                    resource_counts = self._apply_get_resource_counts(events)
                    node = Node2(arr_joined, events, resource_counts)
                    self._nodes.append(node)
            '''
            # Solution 1.2: use matrix multiplication to enumerate combinations
            # TODO: optimize the creation of nodes
            # use batch processing to avoid memory overuse
            product_combs = product(*comb)
            print('Initialize using method `{}`... '.format(self.init_method), end='')
            print('Testing {:,} combinations'.format(total_comb_size))
            tested_comb_size = 0
            n_events_covered = 0
            while True:
                #print('start to construct list of arrays')
                #all_arr_joined = [np.concatenate(prod) for prod in product(*comb)]
                batch_product_combs = list(islice(product_combs, self.init_batch))
                if len(batch_product_combs) == 0:
                    return
                else:
                    tested_comb_size += len(batch_product_combs)

                all_arr_joined = [np.concatenate(prod) for prod in batch_product_combs]
                #print(len(all_arr_joined))
                #print('start to stack arrays')
                all_arr_joined = np.stack(all_arr_joined)
                #print(all_arr_joined)
                #print('nodes matrix shape: {}'.format(all_arr_joined.shape))
                #print('start to locate events')
                #print(self._log)
                #print('log matrix shape: {}'.format(self._log.shape))
                #print('start to do dot product')
                mask = np.matmul(self._log, all_arr_joined.T, dtype=int) == self._n_tda
                #print(mask)
                #print('result matrix shape: {}'.format(mask.shape))
                #print(np.unique(np.nonzero(mask)[0]))
                #print(np.unique(np.nonzero(mask)[1]))
                for j in np.unique(np.nonzero(mask)[1]):
                    arr_joined = all_arr_joined[j,:]
                    events = np.nonzero(mask[:,j])[0]
                    resource_counts = self._apply_get_resource_counts(events)
                    node = Node2(arr_joined, events, resource_counts)
                    self._nodes.append(node)
                    n_events_covered += len(events)
                print('\t{:.0%} tested, covering {:.0%} events.'.format(
                    tested_comb_size / total_comb_size,
                    n_events_covered / self._n_E
                ))
                if n_events_covered >= self._n_E:
                    # stop early if all events have been covered
                    break
        print('Initialization completed with method "{}".'.format(self.init_method))
    
    def _hash_state(self, pars):
        id_state = []
        for tda in (self._tda_case + self._tda_act + self._tda_time):
            arr_int_from_bin = np.dot(
                np.array([2 ** x for x in range(len(pars[tda]))]),
                pars[tda]
            )
            # NOTE: numpy.unique() returns unique elements in sorted order
            id_state.append(np.unique(arr_int_from_bin).tobytes()) 
        return b''.join(id_state)

    def _verify_state(self, nodes):
        n_events = 0
        for i in range(len(nodes)):
            n_events += len(nodes[i].events)
            if set(nodes[i].events) == set(self._apply_to_all(nodes[i].arr)):
                pass
            else:
                raise ValueError('Node array does not match stored events!')
            for j in range(i + 1, len(nodes) - 1):
                if len(set(nodes[i].events).intersection(set(nodes[j].events))) > 0:
                    raise ValueError('Stored events are not disjoint!')
        if n_events != self._n_E:
            raise ValueError('Stored events do not add up to all events!')

    def _apply_to_part(self, arr, n_attrs, rows=None, cols=None):
        # result: |E[rows,cols]| x 1
        # NOTE: the result mask is local to the given rows and columns
        if rows is None:
            mask = np.matmul(self._log[:,cols], arr.T, dtype=int) == n_attrs
        elif cols is None:
            mask = np.matmul(self._log[rows,:], arr.T, dtype=int) == n_attrs
        else:
            mask = np.matmul(self._log[rows,cols], arr.T, dtype=int) == n_attrs

        if np.any(mask):
            return np.nonzero(mask)[0]
        else:
            return []
    
    def _apply_to_all(self, arr):
        # result: |E| x 1
        mask = np.matmul(self._log, arr.T, dtype=int) == self._n_tda
        if np.any(mask):
            return np.nonzero(mask)[0]
        else:
            return []
    
    def _apply_get_resource_counts(self, events):
        uniq_res, counts = np.unique(self._arr_event_r[events], return_counts=True)
        resource_counts = dict(zip(uniq_res, counts))
        return resource_counts
    
    def _neighbors(self):
        '''
        Must return a list of 4-tuples:
            [(attr_name, par, [input part arr], [output part arr])]
        '''
        raise NotImplementedError('Abstract class. Use a child class implementing this method.')
    
    def _neighbor_split(self):
        '''
        Must return a 4-tuple:
            (attr_name, par, [input part arr], [output part arr]) 
                                len=1               len=2 
        '''
        # select an attribute randomly
        attr = self._rng.choice(list(self._tdav.keys()))
        # create a copy
        par = self._pars[attr].copy()
        if np.all(np.any(par, axis=0)) and np.sum(par) == len(par):
            # all singletons, no further split allowed 
            # (i.e., an I matrix after rearrangement)
            return None
        cols_nonzero = np.unique(np.nonzero(par)[1])
        first_col_allzero = np.amin(np.nonzero(np.all(par == 0, axis=0))[0])
        # select an existing, non-singleton part randomly
        self._rng.shuffle(cols_nonzero)
        for col in cols_nonzero:
            if np.sum(par[:,col]) > 1:
                rows_nonzero = np.nonzero(par[:,col])[0]
                # select a separator position randomly
                i_bar = self._rng.choice(len(rows_nonzero) - 1) + 1
                original_col = par[:,col].copy()
                par[rows_nonzero[i_bar:],col] = 0
                par[rows_nonzero[i_bar:],first_col_allzero] = 1
                return tuple((
                    attr, par,
                    [original_col],
                    [par[:,col], par[:,first_col_allzero]]
                ))
    
    def _generate_nodes_split(self, attr, original_cols, new_cols):
        # split: 1 original -> 2 new
        nodes_next = []
        # Solution 2: selectively update current node list via copying
        attr_dim = self._tda_dim[attr]
        if attr_dim == 'CT':
            attr_abs_index = tuple((
                self._index_tda_case[attr][0] + self._index_case[0],
                self._index_tda_case[attr][1] + self._index_case[0],
            ))
        elif attr_dim == 'AT':
            attr_abs_index = tuple((
                self._index_tda_act[attr][0] + self._index_act[0],
                self._index_tda_act[attr][1] + self._index_act[0],
            ))
        else:
            attr_abs_index = tuple((
                self._index_tda_time[attr][0] + self._index_time[0],
                self._index_tda_time[attr][1] + self._index_time[0],
            ))

        # stack the arrays of nodes
        all_arr_joined = np.stack([node.arr for node in self._nodes])

        # generate nodes from split
        is_node_to_split = np.dot(
            all_arr_joined[:,slice(*attr_abs_index)],
            original_cols[0].T
        )
        for i, node in enumerate(self._nodes):
            if is_node_to_split[i]:
                arr_left = node.arr.copy()
                arr_left[slice(*attr_abs_index)] = new_cols[0]
                events_left = node.events[self._apply_to_part(
                    arr=new_cols[0], n_attrs=1, 
                    rows=node.events, cols=slice(*attr_abs_index)
                )]
                if len(events_left) > 0:
                    rc_left = self._apply_get_resource_counts(events_left)
                    nodes_next.append(Node2(arr_left, events_left, rc_left))
                arr_right = node.arr.copy()
                arr_right[slice(*attr_abs_index)] = new_cols[1]
                events_right = node.events[self._apply_to_part(
                    arr=new_cols[1], n_attrs=1, 
                    rows=node.events, cols=slice(*attr_abs_index)
                )]
                if len(events_right) > 0:
                    rc_right = self._apply_get_resource_counts(events_right)
                    nodes_next.append(Node2(arr_right, events_right, rc_right))
            else:
                nodes_next.append(
                    Node2(node.arr, node.events, node.resource_counts.copy())
                )
        return nodes_next

    def _neighbor_merge(self):
        '''
        Must return a 4-tuple:
            (attr_name, par, [input part arr], [output part arr]) 
                                len=2               len=1 
        '''
        # select an attribute randomly
        attr = self._rng.choice(list(self._tdav.keys()))
        par = self._pars[attr].copy()
        if np.sum(np.all(par, axis=0)) == 1:
            # only one singleton, no further merge allowed
            return None
        cols_nonzero = np.unique(np.nonzero(par)[1])
        if len(cols_nonzero) > 2:
            # select two parts randomly, if there exist more than two parts
            sel_cols_nonzero = cols_nonzero[self._rng.choice(len(cols_nonzero), size=2, replace=False)]
        else:
            # merge the only two parts
            sel_cols_nonzero = cols_nonzero[:]
        original_col_left = par[:,sel_cols_nonzero[0]].copy()
        original_col_right = par[:,sel_cols_nonzero[1]].copy()
        # NOTE: for two boolean arrays, plus (+) refers to logical OR
        merged_col = par[:,sel_cols_nonzero[0]] + par[:,sel_cols_nonzero[1]]
        par[:,sel_cols_nonzero[0]] = merged_col
        par[:,sel_cols_nonzero[1]] = 0
        return tuple((
            attr, par,
            [original_col_left, original_col_right],
            [merged_col]
        ))
    
    def _generate_nodes_merge(self, attr, original_cols, new_cols):
        # merge: 2 original -> 1 new
        nodes_next = []
        # Solution 2: selectively update current node list via copying
        attr_dim = self._tda_dim[attr]
        if attr_dim == 'CT':
            attr_abs_index = tuple((
                self._index_tda_case[attr][0] + self._index_case[0],
                self._index_tda_case[attr][1] + self._index_case[0],
            ))
        elif attr_dim == 'AT':
            attr_abs_index = tuple((
                self._index_tda_act[attr][0] + self._index_act[0],
                self._index_tda_act[attr][1] + self._index_act[0],
            ))
        else:
            attr_abs_index = tuple((
                self._index_tda_time[attr][0] + self._index_time[0],
                self._index_tda_time[attr][1] + self._index_time[0],
            ))

        # stack the arrays of nodes
        all_arr_joined = np.stack([node.arr for node in self._nodes])

        # generate nodes from merge
        is_node_to_merge = np.any(np.dot(
            all_arr_joined[:,slice(*attr_abs_index)],
            np.array(original_cols).T
        ), axis=1)
        arr_visited = dict()
        for i, node in enumerate(self._nodes):
            if is_node_to_merge[i]:
                # extract the pattern (in bytes, so hashable)
                # i.e., node.arr excluding the slice being tested
                patt = np.hstack(np.split(node.arr, attr_abs_index)[::2]).tobytes()
                if patt in arr_visited:
                    # create a new node combining data from:
                    #   self._nodes[arr_visited[pattern]] and
                    #   node (the current one)
                    node_to_pair = self._nodes[arr_visited[patt]]
                    events_union = np.union1d(node_to_pair.events, node.events)
                    # append created node to nodes_next
                    # NOTE: for two boolean arrays, plus (+) refers to logical OR
                    nodes_next.append(
                        Node2(
                            node_to_pair.arr + node.arr,
                            events_union,
                            self._apply_get_resource_counts(events_union)
                        )
                    )
                    # delete pattern
                    del arr_visited[patt]
                else:
                    # save node index for pairing
                    arr_visited[patt] = i
            else:
                nodes_next.append(
                    Node2(node.arr, node.events, node.resource_counts.copy())
                )
        # include solo nodes, i.e., nodes unpaired
        for i in arr_visited.values():
            new_arr = self._nodes[i].arr.copy()
            new_arr[slice(*attr_abs_index)] = new_cols[0]
            nodes_next.append(
                Node2(new_arr, self._nodes[i].events, self._nodes[i].resource_counts.copy())
            )
        return nodes_next
    
    def _dist_from_zero(self, pars, normalized=True):
        curr_n_moves = np.sum(
            np.count_nonzero(np.any(par, axis=0)) for par in pars.values()
        ) 
        if normalized:
            return curr_n_moves / self._search_diameter
        else:
            return curr_n_moves

    def _evaluate(self, nodes, pars):
        '''
            Energy function: Combine dispersal and impurity
        '''
        dis = self._calculate_dispersal(nodes)
        imp = self._calculate_impurity(nodes)

        # arithmetic mean
        e = 0.5 * (dis + imp)
        # harmonic mean (with two extremes disabled)
        #e = 2 * dis * imp / (dis + imp)

        return e, dis, imp

    def _calculate_dispersal(self, nodes):
		# calculate the pairwise execution context distance using the "tda_*_par" sequences
        arr_nodes = np.stack([node.arr for node in nodes]) 
        pdist_nodes_case = (squareform(
            pdist(arr_nodes[:,slice(*self._index_case)], metric='hamming')
        ) != 0).astype(np.int_)
        pdist_nodes_act = (squareform(
            pdist(arr_nodes[:,slice(*self._index_act)], metric='hamming')
        ) != 0).astype(np.int_)
        pdist_nodes_time = (squareform(
            pdist(arr_nodes[:,slice(*self._index_time)], metric='hamming')
        ) != 0).astype(np.int_)
        # number of dimensions used can be determined 
        num_dims_used = (
            int(np.any(pdist_nodes_case)) + 
            int(np.any(pdist_nodes_act)) +
            int(np.any(pdist_nodes_time))
        )
        if num_dims_used == 0:
            return 0

        pdist_nodes = (
            (pdist_nodes_case + pdist_nodes_act + pdist_nodes_time)
            / num_dims_used
        )
        # retrieve the contained resources and their counts
        dist_r = defaultdict(lambda: 0)
        # enumerate distinct pairs of Node objects
        for i, j in combinations(range(len(nodes)), r=2):
            # find the intersection of the sets
            for res in (nodes[i].resource_counts.keys() & nodes[j].resource_counts.keys()):
                # the sum of pairwise event distance
                dist_r[res] += (
                    pdist_nodes[i,j] *
                    nodes[i].resource_counts[res] * nodes[j].resource_counts[res]
                )
        # weight by p_r and normalize by C(E_r, 2)
        for res in dist_r.keys():
            dist_r[res] *= self._p_res[res] * (1 / self._ncomb_E_res_2[res])
        return sum(dist_r.values())

    def _calculate_impurity(self, nodes):
        sum_impurity = 0
        # enumerate Node objects
        for node in nodes:
		    # retrieve the contained resources and their counts
            local_p_res = np.array(list(node.resource_counts.values()), dtype=np.double)
            # calculate the ratio of resource-events 
            ratio_res_events = np.sum(local_p_res) / self._n_E
            # calculate local impurity
            sum_impurity += ratio_res_events * scipy_entropy(local_p_res, base=2)
        return sum_impurity / self._max_impurity
    
    def _calculate_sparsity(self, nodes, pars):
        n_pars_comb = 1
        for attr in pars.keys():
            n_pars_comb *= len(np.unique(np.nonzero(pars[attr])[1]))
        return (1.0 - len(nodes) / n_pars_comb)
    
    def _search(self):
        raise NotImplementedError('Abstract class. Use a child class implementing this method.')
    
    def _save_history(self, data_history, columns):
        fnout = 'SearchMiner_{}_stats.out'.format(pd.Timestamp.now().strftime('%Y%m%d-%H%M%S'))
        df_history = pd.DataFrame(
            data=data_history, 
            columns=columns
        )
        df_history.to_csv(
            fnout,
            sep=',',
            na_rep='None',
            float_format='%.6f',
            index=False
        )
        print('Procedure history has been written to file "{}".'.format(fnout))


#########################################################################

class GreedySearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        neighbor_sample_size=10, always_move=False, n_max_move=1000
    ):
        # Initialize system parameters
        self.neighbor_sample_size = neighbor_sample_size
        self.always_move = always_move
        self.n_max_move = n_max_move

        super().__init__(
            el=el, attr_spec=attr_spec, 
            random_number_generator=random_number_generator, 
            init_method=init_method, init_batch=init_batch,
            print_steps=print_steps, trace_history=trace_history
        )
    
    def _neighbors(self):
        # Greedy search generates a few neighbors per iteration
        size = 0
        neighbors = []
        pr_split = 0.5
        while size < self.neighbor_sample_size:
            if self._rng.random() < pr_split:
                n = self._neighbor_split()
                if n is not None:
                    neighbors.append((n, 'split'))
                    size += 1
            else:
                n = self._neighbor_merge()
                if n is not None:
                    neighbors.append((n, 'merge'))
                    size += 1
        return neighbors
    
    def _decide_move(self, E_next, E):
        if self.always_move:
            return True
        else:
            return E_next < E
    
    def _search(self):
        print('Start greedy search with neighbor_sample_size={}, always_move={}, n_max_move={}'.format(self.neighbor_sample_size, self.always_move, self.n_max_move))

        #self._verify_state(self._nodes)

        ret = self._evaluate(self._nodes, self._pars)
        E, dis, imp = ret[0], ret[1], ret[2]

        # keep track of the best state
        step_best = 0
        E_best, dis_best, imp_best = E, dis, imp
        nodes_best = self._nodes.copy()

        k = 0
        # keep track of history, if required
        if self.trace_history:
            # Step, Action, Attribute, 
            # Dispersal, Impurity, Energy
            history = [(0, self.init_method, None, dis, imp, E)]

        while k < self.n_max_move:
            k += 1
            l_neighbors = self._neighbors()
            i_bn = -1
            E_bn, dis_bn, imp_bn = np.inf, np.inf, np.inf
            nodes_neighbor = []
            for i, neighbor in enumerate(l_neighbors):
                move, action = neighbor
                if move is None:
                    pass
                else:
                    attr, new_par, original_cols, new_cols = move
                    if action == 'split':
                        nodes_neighbor = self._generate_nodes_split(attr, original_cols, new_cols)
                    else:
                        nodes_neighbor = self._generate_nodes_merge(attr, original_cols, new_cols)
                    new_pars = self._pars.copy()
                    new_pars[attr] = new_par
                    ret_next = self._evaluate(nodes_neighbor, new_pars)
                    self._verify_state(nodes_neighbor)
                    E_next, dis_next, imp_next = ret_next[0], ret_next[1], ret_next[2]

                    if E_next < E_bn:
                        E_bn = E_next
                        i_bn = i
                        dis_bn = dis_next
                        imp_bn = imp_next
            # keep the best neighbor (bn) for testing
            move_bn, action_bn = l_neighbors[i_bn]
            if action_bn == 'split':
                nodes_next = self._generate_nodes_split(move_bn[0], move_bn[2], move_bn[3])
            else:
                nodes_next = self._generate_nodes_merge(move_bn[0], move_bn[2], move_bn[3])
            #self._verify_state(nodes_next)
            E_next, dis_next, imp_next = E_bn, dis_bn, imp_bn

            if self.print_steps:
                print(f'Step [{k}]\tpropose "{action}" on `{move[0]}`')
                print('\tCurrent #nodes: {}'.format(len(self._nodes)))
                print('\tCurrent energy: {:.6f}'.format(E))
                print('\t\t> Current dispersal: {:.6f}'.format(dis))
                print('\t\t> Current impurity: {:.6f}'.format(imp))
                print('\t' + '-' * 40)
                print('\tNeighbor #nodes: {}'.format(len(nodes_next)))
                print('\tNeighbor energy: {:.6f}'.format(E_next))
                print('\t\t> Neighbor dispersal: {:.6f}'.format(dis_next))
                print('\t\t> Neighbor impurity: {:.6f}'.format(imp_next))
                print('\t' + '-' * 40)
                print('\tBest state @ step [{}]'.format(step_best))
                print('\tBest state #nodes: {}'.format(len(nodes_best)))
                print('\tBest state energy: {:.6f}'.format(E_best))
                print('\t\t> Best state dispersal: {:.6f}'.format(dis_best))
                print('\t\t> Best state impurity: {:.6f}'.format(imp_best))

            # decide whether to move to neighbor
            if self._decide_move(E_next, E):
                # move to neighbor state; update
                if self.print_steps:
                    print('\t\t\t>>> MOVE TO NEIGHBOR')
                self._pars[move_bn[0]] = move_bn[1]
                del self._nodes[:]
                self._nodes = nodes_next
                E, dis, imp = E_next, dis_next, imp_next 

                # check if better than best state
                if E < E_best:
                    step_best = k
                    E_best, dis_best, imp_best = E, dis, imp
                    del nodes_best[:]
                    nodes_best = self._nodes.copy()

            if self.trace_history:
                # Step, Action, Attribute, 
                # Dispersal, Impurity, Energy
                history.append((
                    k, None if move_bn is None else action_bn, None if move_bn is None else move_bn[0],
                    dis, imp, E
                ))
        
        print(f'\nSearch ended at step:\t{k}')
        del self._nodes[:]
        self._nodes = nodes_best
        print('Select best state at step [{}] with:'.format(step_best))
        print('\t #Nodes:\t{}'.format(len(self._nodes)))
        print('\t Energy:\t{:.6f}'.format(E_best))
        print('\t\t> dispersal:\t{:.6f}'.format(dis_best))
        print('\t\t> impurity:\t{:.6f}'.format(imp_best))

        # output history
        if self.trace_history:
            self._save_history(
                history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'dispersal', 'impurity', 'energy'
                ]
            )


class SASearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        neighbor_sample_size=10, T0=1000, Tmin=1, alpha=1
    ):
        # Initialize system parameters
        # initialization method
        self.init_method = init_method
        # batch size to test at initialization, subject to mem size
        self.init_batch = init_batch
        # size of neighborhood
        self.neighbor_sample_size = neighbor_sample_size
        # system initial temperature
        self.T0 = T0
        # minimum temperature allowed
        self.Tmin = Tmin
        # set rate for reducing system temperature
        self.alpha = alpha

        # Set flags
        # whether to print the search procedure
        self.print_steps = print_steps
        # whether to record the history of intermediate stats
        self.trace_history = trace_history

        super().__init__(
            el=el, attr_spec=attr_spec, 
            random_number_generator=random_number_generator, 
            init_method=init_method, init_batch=init_batch,
            print_steps=print_steps, trace_history=trace_history
        )

    def _neighbors(self):
        size = 0
        neighbors = []
        pr_split = 0.5
        while size < self.neighbor_sample_size:
            if self._rng.random() < pr_split:
                n = self._neighbor_split()
                if n is not None:
                    neighbors.append((n, 'split'))
                    size += 1
            else:
                n = self._neighbor_merge()
                if n is not None:
                    neighbors.append((n, 'merge'))
                    size += 1
        return neighbors

    def _prob_acceptance(self, E, E_next, T, 
        dis=None, dis_next=None, imp=None, imp_next=None, 
        E_best=None, dis_best=None, imp_best=None,
        **kwarg
        ):
        # NOTE: allow pr > 1 if E_next < E, to avoid if clause for capping
        # neighbor comparison
        pr = np.exp(-1 * self._n_E * (E_next - E) / T)
        '''
        # neighbor comparison
        if dis_next - dis > 0:
            cost_dis = np.log(dis_next - dis + 1)
        else:
            cost_dis = -1 * np.log(dis - dis_next + 1)
        cost_imp = (imp_next - imp)
        pr = np.exp(-1 * self._n_E * self.T0 * (cost_dis + cost_imp) / T)
        '''
        return pr
    
    def _cooling(self, k):
        return self.T0 - self.alpha * k

    def _search(self):
        print('Start simulated annealing search with T0={}, Tmin={}, alpha={}:'.format(self.T0, self.Tmin, self.alpha))

        #self._verify_state(self._nodes)

        T = self.T0
        ret = self._evaluate(self._nodes, self._pars)
        E, dis, imp = ret[0], ret[1], ret[2]

        # keep track of the best state
        step_best = 0
        E_best, dis_best, imp_best = E, dis, imp
        nodes_best = self._nodes.copy()

        k = 0
        # keep track of history, if required
        if self.trace_history:
            # Step, Action, Attribute, 
            # Probability of acceptance, System temperature, 
            # Dispersal, Impurity, Energy
            history = [(0, self.init_method, None, None, T, dis, imp, E)]

        while T > self.Tmin:
            k += 1
            # generate neighbors
            for i, neighbor in enumerate(self._neighbors()):
                move, action = neighbor[0], neighbor[1]
                if move is None:
                    pass
                else:
                    attr, new_par, original_cols, new_cols = move

                    if action == 'split':
                        nodes_next = self._generate_nodes_split(attr, original_cols, new_cols)
                    else:
                        nodes_next = self._generate_nodes_merge(attr, original_cols, new_cols)
                        
                    new_pars = self._pars.copy()
                    new_pars[attr] = new_par
                    ret_next = self._evaluate(nodes_next, new_pars)
                    #self._verify_state(nodes_next)
                    E_next, dis_next, imp_next = ret_next[0], ret_next[1], ret_next[2]

                    if self.print_steps:
                        print(f'Step [{k}]\tpropose "{action}" on `{move[0]}`')
                        print('\tCurrent temperature:\t{:.3f}'.format(T))
                        print('\tCurrent #nodes: {}'.format(len(self._nodes)))
                        print('\tCurrent energy: {:.6f}'.format(E))
                        print('\t\t> Current dispersal: {:.6f}'.format(dis))
                        print('\t\t> Current impurity: {:.6f}'.format(imp))
                        print('\t' + '-' * 40)
                        print('\tNeighbor #nodes: {}'.format(len(nodes_next)))
                        print('\tNeighbor energy: {:.6f}'.format(E_next))
                        print('\t\t> Neighbor dispersal: {:.6f}'.format(dis_next))
                        print('\t\t> Neighbor impurity: {:.6f}'.format(imp_next))
                        print('\t' + '-' * 40)
                        print('\tBest state @ step [{}]'.format(step_best))
                        print('\tBest state #nodes: {}'.format(len(nodes_best)))
                        print('\tBest state energy: {:.6f}'.format(E_best))
                        print('\t\t> Best state dispersal: {:.6f}'.format(dis_best))
                        print('\t\t> Best state impurity: {:.6f}'.format(imp_best))

                    # decide whether to move to neighbor
                    prob_acceptance = self._prob_acceptance(
                        E=E, E_next=E_next, T=T, 
                        dis=dis, dis_next=dis_next, imp=imp, imp_next=imp_next, 
                        E_best=E_best, dis_best=dis_best, imp_best=imp_best
                    )
                    if self.print_steps:
                        print('\tProbability of moving: {}'.format(prob_acceptance))

                    if self._rng.random() < prob_acceptance:
                        # move to neighbor state; update
                        if self.print_steps:
                            print('\t\t\t>>> MOVE TO NEIGHBOR')
                        self._pars[attr] = new_par
                        del self._nodes[:]
                        self._nodes = nodes_next
                        E, dis, imp = E_next, dis_next, imp_next 

                        # check if better than best state
                        if E < E_best:
                            step_best = k
                            E_best, dis_best, imp_best = E, dis, imp
                            del nodes_best[:]
                            nodes_best = self._nodes.copy()
                    else:
                        pass

            if self.trace_history:
                # Step, Action, Attribute, 
                # Probability of acceptance, System temperature, 
                # Dispersal, Impurity, Energy
                history.append((
                    k, None if move is None else action, None if move is None else move[0],
                    0.0 if move is None else prob_acceptance, T,
                    dis, imp, E
                ))
            # cool down system temperature
            T = self._cooling(k)
        
        print(f'\nSearch ended with final system temperature:\t{T}')
        del self._nodes[:]
        self._nodes = nodes_best
        print('Select best state @ step [{}] with:'.format(step_best))
        print('\t #Nodes:\t{}'.format(len(self._nodes)))
        print('\t Energy:\t{:.6f}'.format(E_best))
        print('\t\t> dispersal:\t{:.6f}'.format(dis_best))
        print('\t\t> impurity:\t{:.6f}'.format(imp_best))

        # output history
        if self.trace_history:
            self._save_history(
                history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'prob_acceptance', 'temp',
                    'dispersal', 'impurity', 'energy'
                ]
            )
