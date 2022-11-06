from collections import defaultdict
from itertools import product, combinations
from math import comb as num_combinations

from datetime import datetime

import numpy as np
from pandas import get_dummies as apply_ohe
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import pdist, squareform

from ordinor.utils.validation import check_convert_input_log
import ordinor.exceptions as exc
import ordinor.constants as const

from ordinor.execution_context.base import BaseMiner

from .TreeNode import Node2
from .AtomicRule import AtomicRule
from .Rule import Rule

class SearchMiner(BaseMiner):
    def __init__(self, 
        el, attr_spec, 
        temp_init=None,
        temp_min=None,
        random_number_generator=None
    ):
        # Parse log
        if el is not None:
            self._log = check_convert_input_log(el).copy()
            # only use events with resource information
            self._log = self._log[self._log[const.RESOURCE].notna()]
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
                if spec['attr_type'] == 'categorical':
                    if spec['attr_dim'] == 'CT':
                        self._tda_case.append(attr)
                    elif spec['attr_dim'] == 'AT':
                        self._tda_act.append(attr)
                    else:
                        self._tda_time.append(attr)
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
            self._par = dict()
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
            # NOTE: |E|, log size (after exclusion)
            self._n_E = len(self._log)
            # NOTE: absolute frequencies of resources |E_r|
            #       (over all events with resource information)
            self._p_res = self._log[const.RESOURCE].value_counts(
                normalize=False, sort=False, dropna=True
            ).to_dict()
            # NOTE: integers, to be used for calculating dispersal
            self._ncomb_E_res_2 = dict(
                (r, num_combinations(count_r, 2)) for r, count_r in self._p_res.items()
            )
            # normalize the frequencies of resources
            sum_n_E_res = sum(self._p_res.values())
            for r in self._p_res.keys():
                self._p_res[r] /= sum_n_E_res
            # NOTE: float constant, to be used for calculating impurity
            self._max_impurity = scipy_entropy(
                np.array(list(self._p_res.values()), dtype=np.double), base=2
            )

            # NOTE: an |E| x 1 array recording originator resources
            self._arr_event_r = self._log[const.RESOURCE].to_numpy(copy=True, na_value='/')
            # NOTE: |E_res|, number of events with resource information
            self._n_E_res = self._n_E - np.sum(self._arr_event_r == '/')

            # drop non-type-defining attributes columns
            self._log.drop(columns=included_cols, inplace=True)
            # apply One-Hot-Encoding to the log (pandas.get_dummies)
            # NOTE: use consistent prefix separator
            self._ohe_prefix_sep = '^^@^^'
            self._log = apply_ohe(
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
            self._log = self._log.to_numpy(dtype=np.uintc)

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

            # Initialize temperature for searching
            self.T0 = temp_init
            self.Tmin = temp_min

            # TODO: Trigger search
            # TODO: model user-supplied categorization rules
            self._search()

            super().__init__(el)
        else:
            raise ValueError('invalid attribute specification')

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
            arr_case = np.array(k_arr_case, dtype=np.uintc)
            ct_label = i
            # translate to rules and save
            ct_rules = []
            for attr in self._tda_case:
                arr_case_attr = arr_case[slice(*self._index_tda_case[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_case_attr)[0])
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
            arr_act = np.array(k_arr_act, dtype=np.uintc)
            at_label = i
            # translate to rules and save
            at_rules = []
            for attr in self._tda_act:
                arr_act_attr = arr_act[slice(*self._index_tda_act[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_act_attr)[0])
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
            arr_time = np.array(k_arr_time, dtype=np.uintc)
            tt_label = i
            # translate to rules and save
            tt_rules = []
            for attr in self._tda_time:
                arr_time_attr = arr_time[slice(*self._index_tda_time[attr])]
                sel_attr_values = frozenset(self._tdav[attr][i] for i in np.nonzero(arr_time_attr)[0])
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

    def _init_state(self, init_method='zero', **kwargs):
        # initialize partitions on all attributes
        # initialize nodes correspondingly
        comb = []
        for i, tda in enumerate([self._tda_case, self._tda_act, self._tda_time]):
            tda_i = []
            comb_i = []
            if len(tda) == 0:
                continue
            for attr in tda:
                attr_parts = []
                n = len(self._tdav[attr])
                if n == 1:
                    self._par[attr] = np.array([[1]], dtype=np.uintc)
                    attr_parts.append(np.array([1], dtype=np.uintc))
                else:
                    if init_method == 'zero':
                        self._par[attr] = np.zeros((n, n), dtype=np.uintc)
                        # apply no partition to all attributes (one-holds-all)
                        self._par[attr][:,0] = 1
                        attr_parts.append(self._par[attr][:,0])
                    elif init_method == 'full_split':
                        # create a singleton for each value
                        self._par[attr] = np.eye(n, dtype=np.uintc)
                        for j in range(n):
                            attr_parts.append(self._par[attr][:,j])
                    elif init_method == 'random':
                        self._par[attr] = np.zeros((n, n), dtype=np.uintc)
                        # select a part for each attribute value randomly
                        for i in range(n):
                            i_part = self._rng.choice(n)
                            self._par[attr][i,i_part] = 1
                        for j in range(n):
                            if self._par[attr][:,j].any():
                                attr_parts.append(self._par[attr][:,j])
                    else:
                        # TODO: model user-supplied categorization rules
                        # init_method == 'informed'
                        raise NotImplementedError
                tda_i.append(attr_parts)
            for prod in product(*tda_i):
                comb_i.append(np.concatenate(prod))
            comb.append(comb_i)
        
        if init_method == 'full_split':
            # TODO: need to start from existing combinations instead of enumeration
            raise NotImplementedError
        else:
            # TODO: optimize the creation of nodes
            # TODO: verify
            print('start to construct list of arrays')
            all_arr_joined = [np.concatenate(prod) for prod in product(*comb)]
            print(len(all_arr_joined))
            print('start to stack arrays')
            all_arr_joined = np.stack(all_arr_joined)
            print(all_arr_joined.shape)
            print('start to locate events')
            print(self._log.shape)
            print(self._log.dtype)
            print(all_arr_joined.T.shape)
            print(all_arr_joined.T.dtype)
            print('start to do dot product')
            mask = np.matmul(self._log, all_arr_joined.T) 
            print(mask.shape)
            print('start to do value comparison')
            mask = mask == self._n_tda
            print(mask.shape)
            exit(1)
            for j in np.unique(np.nonzero(mask)[1]):
                arr_joined = all_arr_joined[j,:]
                events = np.nonzero(mask[:,j])[0]
                resource_counts = self._apply_get_resource_counts(events)
                node = Node2(arr_joined, events, resource_counts)
                self._nodes.append(node)
            '''
            for prod in product(*comb):
                arr_joined = np.concatenate(prod)
                events = self._apply_to_all(arr_joined)
                if len(events) > 0:
                    resource_counts = self._apply_get_resource_counts(events)
                    node = Node2(arr_joined, events, resource_counts)
                    self._nodes.append(node)
            '''
        
    def _apply_to_part(self, arr, n_attrs, rows=None, cols=None):
        # res: |E| x 1
        if rows is None:
            mask = np.matmul(self._log[:,cols], arr.T) == n_attrs
        elif cols is None:
            mask = np.matmul(self._log[rows,:], arr.T) == n_attrs
        else:
            mask = np.matmul(self._log[rows,cols], arr.T) == n_attrs

        if np.any(mask):
            return np.nonzero(mask)[0]
        else:
            return []
    
    def _apply_to_all(self, arr):
        # res: |E| x 1
        mask = np.matmul(self._log, arr.T) == self._n_tda
        if np.any(mask):
            return np.nonzero(mask)[0]
        else:
            return []
    
    def _apply_get_resource_counts(self, events):
        uniq_res, counts = np.unique(self._arr_event_r[events], return_counts=True)
        resource_counts = dict(zip(uniq_res, counts))
        return resource_counts

    def _neighbor(self):
        '''
            Must return a 4-tuple, or None:
                (attr_name, par, [input part arr], [output part arr]) 
        '''
        i = self._rng.choice(2)
        if i == 0:
            return self._neighbor_split(), 'split'
        elif i == 1:
            return self._neighbor_merge(), 'merge'
        else:
            raise ValueError('undefined move to neighboring states')

    def _neighbor_split(self):
        '''
            Must return a 4-tuple, or None:
                (attr_name, par, [input part arr], [output part arr]) 
                                    len=1               len=2 
        '''
        # select an attribute randomly
        attr = self._rng.choice(list(self._tdav.keys()))
        # create a copy
        par = self._par[attr].copy()
        if np.all(np.any(par, axis=0)) and np.sum(par) == len(par):
            # all singletons, no further split allowed
            return None
        cols_nonzero = np.unique(np.nonzero(par)[1])
        first_col_allzero = np.amin(np.nonzero(np.all(par == 0, axis=0))[0])
        self._rng.shuffle(cols_nonzero)
        for col in cols_nonzero:
            if np.sum(par[:,col]) > 1:
                rows_nonzero = np.nonzero(par[:,col])[0]
                i_bar = self._rng.choice(len(rows_nonzero) - 1) + 1
                original_col = par[:,col].copy()
                par[rows_nonzero[i_bar:],col] = 0
                par[rows_nonzero[i_bar:],first_col_allzero] = 1
                '''
                return (
                    attr, 
                    par,
                    [original_col],
                    [par[:,col], par[:,first_col_allzero]]
                )
                '''
                return (
                    attr, par
                )

    def _neighbor_merge(self):
        '''
            Must return a 4-tuple, or None:
                (attr_name, par, [input part arr], [output part arr]) 
                                    len=2               len=1 
        '''
        # select an attribute randomly
        attr = self._rng.choice(list(self._tdav.keys()))
        par = self._par[attr].copy()
        if np.sum(np.all(par, axis=0)) == 1:
            # only singleton, no further merge allowed
            return None
        cols_nonzero = np.unique(np.nonzero(par)[1])
        if len(cols_nonzero) > 2:
            sel_cols_nonzero = cols_nonzero[self._rng.choice(len(cols_nonzero), size=2, replace=False)]
        else:
            sel_cols_nonzero = cols_nonzero[:]
        original_col_left = par[:,sel_cols_nonzero[0]].copy()
        original_col_right = par[:,sel_cols_nonzero[1]].copy()
        merged_col = par[:,sel_cols_nonzero[0]] + par[:,sel_cols_nonzero[1]]
        par[:,sel_cols_nonzero[0]] = merged_col
        par[:,sel_cols_nonzero[1]] = 0
        '''
        return (
            attr,
            par,
            [original_col_left, original_col_right],
            [merged_col]
        )
        '''
        return (attr, par)

    def _evaluate(self, nodes):
        '''
            Energy function: Combine dispersal and impurity
        '''
        dis = self._calculate_dispersal(nodes)
        imp = self._calculate_impurity(nodes)
        # arithmetic mean
        return 0.5 * (dis + imp)
        # harmonic mean
        #return 2 * dis * imp / (dis + imp)

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
            for r in (nodes[i].resource_counts.keys() & nodes[j].resource_counts.keys()):
                # the sum of pairwise event distance
                dist_r[r] += (
                    pdist_nodes[i,j] *
                    nodes[i].resource_counts[r] * nodes[j].resource_counts[r]
                )
        # weight by p_r and normalize by C(E_r, 2)
        for r in dist_r.keys():
            dist_r[r] *= self._p_res[r] * (1 / self._ncomb_E_res_2[r])
        return sum(dist_r.values())

    def _calculate_impurity(self, nodes):
        sum_impurity = 0
        # enumerate Node objects
        for node in nodes:
		    # retrieve the contained resources and their counts
            local_p_res = np.array(list(node.resource_counts.values()), dtype=np.double)
            # calculate the ratio of resource-events 
            ratio_res_events = np.sum(local_p_res) / self._n_E_res
            # calculate local impurity
            sum_impurity += ratio_res_events * scipy_entropy(local_p_res, base=2)
        return sum_impurity / self._max_impurity
    
    def _calculate_efficiency(self, nodes):
        raise NotImplementedError

    def _init_system(self):
        self.T0 = 3000 if self.T0 is None else self.T0
        self.Tmin = 1 if self.Tmin is None else self.Tmin
    
    def _prob_acceptance(self, E, E_next, T, **kwarg):
        pr = np.exp(-1 * 1e4 * (E_next - E) / T)
        #pr = 1 if E_next < E else 0
        return pr
    
    def _cooling(self, T, k):
        alpha = 1
        return self.T0 - alpha * k

    def _search(self):
        self._init_state(init_method='random')
        self._init_system()
        print('Init finished')

        '''
        T = self.T0
        E = self._evaluate(self._nodes)
        E_best = E
        nodes_best = self._nodes.copy()
        k = 0
        while T > self.Tmin:
            k += 1
            move, action = self._neighbor()
            if move is None:
                pass
            else:
                attr, new_par = move[0], move[1]

                # TODO: how to efficiently peek?
                nodes_next = []
                new_pars = self._par.copy()
                new_pars[move[0]] = new_par
                comb = []
                for i, tda in enumerate([self._tda_case, self._tda_act, self._tda_time]):
                    tda_i = []
                    comb_i = []
                    if len(tda) == 0:
                        continue
                    for attr in tda:
                        attr_parts = []
                        n = len(self._tdav[attr])
                        if n == 1:
                            attr_parts.append(np.array([1], dtype=np.uintc))
                        else:
                            for j in np.unique(np.nonzero(new_pars[attr])[1]):
                                attr_parts.append(new_pars[attr][:,j])
                        tda_i.append(attr_parts)
                    for prod in product(*tda_i):
                        comb_i.append(np.concatenate(prod))
                    comb.append(comb_i)

                for prod in product(*comb):
                    arr_joined = np.concatenate(prod)
                    events = self._apply_to_all(arr_joined)
                    if len(events) > 0:
                        resource_counts = self._apply_get_resource_counts(events)
                        node = Node2(arr_joined, events, resource_counts)
                        nodes_next.append(node)
                # TODO end

                E_next = self._evaluate(nodes_next)

                print(f'Step [{k}]\t{action} on {move[0]}')
                print('\tCurrent temperature:\t{:.3f}'.format(T))
                print('\tCurrent energy: {:.6f}'.format(E))
                print('\tCurrent #nodes: {}'.format(len(self._nodes)))
                print('\tNeighbor energy: {:.6f}'.format(E_next))
                print('\tNeighbor #nodes: {}'.format(len(nodes_next)))
                
                # decide move to neighbor
                prob_acceptance = self._prob_acceptance(E, E_next, T)
                print('\tProbability of moving: {}'.format(prob_acceptance))
                if self._rng.random() < prob_acceptance:
                    print('\t\t\t>>> Moved to neighbor')
                    self._par[move[0]] = new_par
                    del self._nodes[:]
                    self._nodes = nodes_next
                    E = E_next
                    if E < E_best:
                        E_best = E
                        del nodes_best[:]
                        nodes_best = nodes_next.copy()
                    
            # cool down system temperature
            T = self._cooling(T, k)
        
        del self._nodes[:]
        self._nodes = nodes_best
        '''

        # TODO: verify the calculation of dispersal and impurity
        start = datetime.now()
        print(self._calculate_dispersal(self._nodes))
        mid = datetime.now()
        print(mid - start)
        start2 = datetime.now()
        print(self._calculate_impurity(self._nodes))
        end = datetime.now()
        print(end - start2)
