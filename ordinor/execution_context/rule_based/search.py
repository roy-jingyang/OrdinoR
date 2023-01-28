'''
A series of search-based solutions:
    - greedy
    - greedy to implement the ODT-based method
    - simulated annealing
'''

import numpy as np
from scipy.stats import median_abs_deviation as mad

from .BaseSearch import BaseSearchMiner

class GreedySearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        size_neighborhood=None, max_iter=1000
    ):
        # Initialize system parameters
        # size of neighborhood per iteration
        self.size_neighborhood = size_neighborhood
        # maximum number of iterations allowed
        self.max_iter = max_iter

        # Initialize additional data structures (tracking visited states)
        self._visited_states = set()
        self._l_dispersal = []
        self._l_impurity = []

        super().__init__(
            el=el, attr_spec=attr_spec, 
            random_number_generator=random_number_generator, 
            init_method=init_method, init_batch=init_batch,
            print_steps=print_steps, trace_history=trace_history
        )
    
    def _neighbors(self):
        # Greedy search generates a few neighbors per iteration
        # NOTE: only use feasible neighbors, i.e., non-"empty"
        neighbors = []
        pr_split = 0.5
        i = 0
        while i < self.size_neighborhood:
            if self._rng.random() < pr_split:
                n = self._neighbor_split()
                action = 'split'
            else:
                n = self._neighbor_merge()
                action = 'merge'
            if n:
                neighbors.append((n, action))
                i += 1
        return neighbors
    
    def _decide_move(
        self, 
        E_next=None, E_curr=None,
    ):
        return (E_next - E_curr) < 0

    def _search(self):
        # determine size of neighborhood, if needed
        if self.size_neighborhood is None:
            self.size_neighborhood = self._n_tda

        print('Start greedy search with size_neighborhood={}, max_iter={}'.format(self.size_neighborhood, self.max_iter))

        #self._verify_state(self._nodes)

        ret = self._evaluate(self._nodes, self._pars)
        E, dis, imp = ret[0], ret[1], ret[2]

        self._visited_states.add(self._hash_state(self._pars))
        self._l_dispersal.append(dis)
        self._l_impurity.append(imp)

        # keep track of the best state 
        # (excl. start state, if init_method is `zero` or `full_split`)
        if self.init_method in ['zero' or 'full_split']:
            id_state_excl = self._hash_state(self._pars)
            step_best = None
            E_best, dis_best, imp_best = None, None, None
            nodes_best = None
        else:
            step_best = 0
            E_best, dis_best, imp_best = E, dis, imp
            nodes_best = self._nodes.copy()

        k = 0
        # keep track of history, if required
        if self.trace_history:
            # Step, Action, Attribute, 
            # Dispersal, Impurity, Energy
            history = [(0, self.init_method, None, dis, imp, E)]

        while k < self.max_iter:
            k += 1
            l_neighbors = self._neighbors()
            # bn: Best Neighbor
            i_bn = None
            E_bn, dis_bn, imp_bn = None, None, None
            nodes_neighbor = []
            for i, neighbor in enumerate(l_neighbors):
                move, action = neighbor
                attr, new_par, original_cols, new_cols = move
                if action == 'split':
                    nodes_neighbor = self._generate_nodes_split(attr, original_cols, new_cols)
                else:
                    nodes_neighbor = self._generate_nodes_merge(attr, original_cols, new_cols)
                new_pars = self._pars.copy()
                new_pars[attr] = new_par
                ret_next = self._evaluate(nodes_neighbor, new_pars)
                #self._verify_state(nodes_neighbor)
                E_next, dis_next, imp_next = ret_next[0], ret_next[1], ret_next[2]

                # record unique states visited
                id_state_next = self._hash_state(new_pars)
                if id_state_next not in self._visited_states:
                    self._visited_states.add(id_state_next)
                    self._l_dispersal.append(dis_next)
                    self._l_impurity.append(imp_next)

                if i_bn is None or self._decide_move(E_next=E_next, E_curr=E_bn):
                    E_bn = E_next
                    i_bn = i
                    dis_bn = dis_next
                    imp_bn = imp_next
            
            # keep only the best neighbor (bn) for testing
            move_bn, action_bn = l_neighbors[i_bn]
            if action_bn == 'split':
                nodes_next = self._generate_nodes_split(move_bn[0], move_bn[2], move_bn[3])
            else:
                nodes_next = self._generate_nodes_merge(move_bn[0], move_bn[2], move_bn[3])
            #self._verify_state(nodes_next)
            E_next, dis_next, imp_next = E_bn, dis_bn, imp_bn

            if self.print_steps:
                print(f'Step [{k}]\tpropose "{action}" on `{move_bn[0]}`')
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
                if E_best:
                    print('\tCurrent best state @ step [{}]'.format(step_best))
                    print('\tCurrent best state #nodes: {}'.format(len(nodes_best)))
                    print('\tCurrent best state energy: {:.6f}'.format(E_best))
                    print('\t\t> Current best state dispersal: {:.6f}'.format(dis_best))
                    print('\t\t> Current best state impurity: {:.6f}'.format(imp_best))

            # decide whether to move to neighbor
            is_moved = False
            if self._decide_move(E_next=E_next, E_curr=E):
                is_moved = True
                # move to best neighboring state; update
                if self.print_steps:
                    print('\t\t\t>>> MOVE TO NEIGHBOR')
                self._pars[move_bn[0]] = move_bn[1]
                del self._nodes[:]
                self._nodes = nodes_next
                E, dis, imp = E_next, dis_next, imp_next 

            # record step history
            if self.trace_history:
                # Step, Action, Attribute, 
                # Dispersal, Impurity, Energy
                history.append((
                    k, action_bn if is_moved else None, move_bn[0] if is_moved else None,
                    dis, imp, E
                ))

            # check if a new best state is found
            has_new_best = (
                E_best is None or 
                E < E_best and self._hash_state(self._pars) != id_state_excl
            )
            if has_new_best:
                step_best = k
                E_best, dis_best, imp_best = E, dis, imp
                if nodes_best:
                    del nodes_best[:]
                nodes_best = self._nodes.copy()

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
                search_history=history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'dispersal', 'impurity', 'energy'
                ],
                l_dispersal=self._l_dispersal, l_impurity=self._l_impurity
            )

class GreedyODTMiner(GreedySearchMiner):
    '''This class implements the ODT-based method using the search framework.
    To do so, configure the Greedy Search as follows:
        * always initialize from the zero state (all events in one cube)
        * neighborhood size is specified per attribute, not per iteration
        * use only the split move (i.e., no backtracking)
        * use `max_iter` as the maximum tree height allowed
    '''
    def __init__(self,
        el, attr_spec,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        size_neighborhood=1,
        max_iter=10
    ):
        # Initialize system parameters
        # size of neighborhood is per attribute, not per iteration
        self.size_neighborhood = size_neighborhood
        # maximum number of iterations allowed
        self.max_iter = max_iter

        # Initialize additional data structures (tracking visited states)
        self._visited_states = set()
        self._l_dispersal = []
        self._l_impurity = []

        super().__init__(
            el=el, attr_spec=attr_spec, 
            random_number_generator=random_number_generator, 
            init_method='zero', init_batch=1,
            print_steps=print_steps, trace_history=trace_history,
            size_neighborhood=self.size_neighborhood, max_iter=self.max_iter
        )
    
    def _neighbors(self):
        # NOTE: only use feasible neighbors, i.e., non-"empty"
        neighbors = []
        # ODT-based method always loops over all attributes
        for attr in self._tdav.keys():
            i = 0
            # size of neighborhood is per attribute, not per iteration
            while i < self.size_neighborhood:
                n = self._neighbor_split(attr=attr)
                action = 'split'
                if n:
                    neighbors.append((n, action))
                    i += 1
                else:
                    # ignore attribute, if fully-split
                    break
        return neighbors

class SASearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        size_neighborhood=None, T0=1000, Tmin=1e-4, alpha=0.99, restart_interval=10,
    ):
        # Initialize system parameters
        # initialization method
        self.init_method = init_method
        # batch size to test at initialization, subject to mem size
        self.init_batch = init_batch
        # size of neighborhood per temperature
        self.size_neighborhood = size_neighborhood
        # system initial temperature
        self.T0 = T0
        # minimum temperature allowed
        self.Tmin = Tmin
        # set rate for reducing system temperature
        self.alpha = alpha
        # set restart interval by number of steps 
        self.restart_interval = restart_interval

        # Initialize additional data structures (tracking visited states)
        self._visited_states = set()
        self._l_dispersal = []
        self._l_impurity = []

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
        # Simulated annealing makes single perturbation sequentially
        # NOTE: only use feasible neighbors, i.e., non-"empty"
        pr_split = 0.5
        while True:
            if self._rng.random() < pr_split:
                n = self._neighbor_split()
                action = 'split'
            else:
                n = self._neighbor_merge()
                action = 'merge'
            if n:
                return n, action

    def _decide_move(
        self, 
        T, E_next=None, E_curr=None,
    ):
        delta_E = E_next - E_curr
        if delta_E <= 0:
            return 1.0
        else:
            return np.exp(-1 * delta_E / T)
    
    def _cooling(self, k):
        # exponential multiplicative cooling 
        return self.T0 * self.alpha ** k

    def _search(self):
        # determine size of neighborhood, if needed
        if self.size_neighborhood is None:
            self.size_neighborhood = self._n_tda

        print('Start simulated annealing search with size_neighborhood={}, T0={}, Tmin={}, alpha={}, restart_interval={}:'.format(self.size_neighborhood, self.T0, self.Tmin, self.alpha, self.restart_interval))

        #self._verify_state(self._nodes)

        T = self.T0
        ret = self._evaluate(self._nodes, self._pars)
        E, dis, imp = ret[0], ret[1], ret[2]

        self._visited_states.add(self._hash_state(self._pars))
        self._l_dispersal.append(dis)
        self._l_impurity.append(imp)

        # keep track of the best state 
        # (excl. start state, if init_method is `zero` or `full_split`)
        if self.init_method in ['zero' or 'full_split']:
            id_state_excl = self._hash_state(self._pars)
            step_best = None
            E_best, dis_best, imp_best = None, None, None
            nodes_best = None
            pars_best = None
        else:
            step_best = 0
            E_best, dis_best, imp_best = E, dis, imp
            nodes_best = self._nodes.copy()
            pars_best = self._pars.copy()

        k = 0
        cnt_restart = 0
        # keep track of history, if required
        if self.trace_history:
            # Step, Action, Attribute, 
            # Probability of acceptance, System temperature, 
            # Dispersal, Impurity, Energy
            history = [(0, self.init_method, None, None, T, dis, imp, E)]

        while T > self.Tmin:
            k += 1
            cnt_restart += 1
            # restart if best has not been updated after specified interval
            if self.restart_interval and cnt_restart == self.restart_interval:
                cnt_restart = 0
                print(f'Restart search at step [{k}], using best state found at step [{step_best}]')
                del self._nodes[:]
                self._nodes = nodes_best.copy()
                del self._pars
                self._pars = pars_best.copy()
                E, dis, imp = E_best, dis_best, imp_best

            # generate neighbors and visit them sequentially
            for i in range(self.size_neighborhood):
                move, action = self._neighbors()
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
                    if E_best:
                        print('\tCurrent best state @ step [{}]'.format(step_best))
                        print('\tCurrent best state #nodes: {}'.format(len(nodes_best)))
                        print('\tCurrent best state energy: {:.6f}'.format(E_best))
                        print('\t\t> Current best state dispersal: {:.6f}'.format(dis_best))
                        print('\t\t> Current best state impurity: {:.6f}'.format(imp_best))

                # record unique states visited
                id_state_next = self._hash_state(new_pars)
                if id_state_next not in self._visited_states:
                    self._visited_states.add(id_state_next)
                    self._l_dispersal.append(dis_next)
                    self._l_impurity.append(imp_next)

                # decide whether to move to neighbor
                prob_acceptance = self._decide_move(
                    T=T, E_next=E_next, E_curr=E
                )
                if self.print_steps:
                    print('\tProbability of moving: {}'.format(prob_acceptance))
                is_moved = False
                if self._rng.random() < prob_acceptance:
                    is_moved = True
                    # move to neighbor state; update
                    if self.print_steps:
                        print('\t\t\t>>> MOVE TO NEIGHBOR')
                    self._pars[attr] = new_par
                    del self._nodes[:]
                    self._nodes = nodes_next
                    E, dis, imp = E_next, dis_next, imp_next 
                
                # record step history
                if self.trace_history:
                    # Step, Action, Attribute, 
                    # Probability of acceptance, System temperature, 
                    # Dispersal, Impurity, Energy
                    history.append((
                        k, action if is_moved else None, move[0] if is_moved else None,
                        prob_acceptance, T,
                        dis, imp, E
                    ))

                # check if a new best state is found
                has_new_best = (
                    E_best is None or 
                    E < E_best and self._hash_state(self._pars) != id_state_excl
                )
                if has_new_best:
                    step_best = k
                    E_best, dis_best, imp_best = E, dis, imp
                    if nodes_best:
                        del nodes_best[:]
                    nodes_best = self._nodes.copy()
                    if pars_best:
                        del pars_best
                    pars_best = self._pars.copy()
                    # reset restart counter
                    cnt_restart = 0

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
                search_history=history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'prob_acceptance', 'temp',
                    'dispersal', 'impurity', 'energy'
                ],
                l_dispersal=self._l_dispersal, l_impurity=self._l_impurity
            )
