'''
A series of search-based solutions:
    - greedy descent
    - simulated annealing
'''

import numpy as np
from scipy.stats import median_abs_deviation as mad

from .BaseSearch import BaseSearchMiner

class RandomWalkSearchMiner(BaseSearchMiner):
    # TODO: for testing purpose
    ...

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
        size = 0
        neighbors = []
        pr_split = 0.5
        # fill in non-empty choices of neighbors
        while size < self.size_neighborhood:
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
    
    def _decide_move(self, 
        E_next=None, E_curr=None,
        dis_next=None, dis_curr=None, imp_next=None, imp_curr=None
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
                    #self._verify_state(nodes_neighbor)
                    E_next, dis_next, imp_next = ret_next[0], ret_next[1], ret_next[2]

                    # record unique states visited
                    id_state_next = self._hash_state(new_pars)
                    if id_state_next not in self._visited_states:
                        self._visited_states.add(id_state_next)
                        self._l_dispersal.append(dis_next)
                        self._l_impurity.append(imp_next)
                    else:
                        pass

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
            if self._decide_move(E_next=E_next, E_curr=E):
                # move to best neighboring state; update
                if self.print_steps:
                    print('\t\t\t>>> MOVE TO NEIGHBOR')
                self._pars[move_bn[0]] = move_bn[1]
                del self._nodes[:]
                self._nodes = nodes_next
                E, dis, imp = E_next, dis_next, imp_next 

                if self.trace_history:
                    # Step, Action, Attribute, 
                    # Dispersal, Impurity, Energy
                    history.append((
                        k, action_bn, move_bn[0],
                        dis, imp, E
                    ))

                # check if better than best state
                has_new_best = E < E_best

                if has_new_best:
                    step_best = k
                    E_best, dis_best, imp_best = E, dis, imp
                    del nodes_best[:]
                    nodes_best = self._nodes.copy()
            else:
                # best neighbor is worse; no move
                if self.trace_history:
                    # Step, Action, Attribute, 
                    # Dispersal, Impurity, Energy
                    history.append((
                        k, None, None,
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
                search_history=history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'dispersal', 'impurity', 'energy'
                ],
                l_dispersal=self._l_dispersal, l_impurity=self._l_impurity
            )


class SASearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        size_neighborhood=None, T0=1000, Tmin=1e-3, alpha=0.95, restart_interval=10,
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
        pr_split = 0.5
        if self._rng.random() < pr_split:
            return self._neighbor_split(), 'split'
        else:
            return self._neighbor_merge(), 'merge'

    def _decide_move(self, 
        T,
        E_next=None, E_curr=None,
        dis_next=None, dis_curr=None, imp_next=None, imp_curr=None, 
        **kwargs
    ):
        delta_E = E_next - E_curr
        if delta_E <= 0:
            return 1.0
        else:
            return np.exp(-1 * delta_E / T)
    
    def _cooling(self, k):
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

                    id_state_next = self._hash_state(new_pars)
                    if id_state_next not in self._visited_states:
                        self._visited_states.add(id_state_next)
                        self._l_dispersal.append(dis_next)
                        self._l_impurity.append(imp_next)
                    else:
                        pass
                        '''
                        # Tabu: prevent moving into visited states
                        continue
                        '''

                    # decide whether to move to neighbor
                    prob_acceptance = self._decide_move(
                        T=T, E_next=E_next, E_curr=E
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
                        has_new_best = E < E_best

                        if has_new_best:
                            step_best = k
                            E_best, dis_best, imp_best = E, dis, imp
                            del nodes_best[:]
                            nodes_best = self._nodes.copy()
                            del pars_best
                            pars_best = self._pars.copy()
                            # reset restart counter
                            cnt_restart = 0
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
                search_history=history, 
                columns=[
                    'step', 'action', 'attribute', 
                    'prob_acceptance', 'temp',
                    'dispersal', 'impurity', 'energy'
                ],
                l_dispersal=self._l_dispersal, l_impurity=self._l_impurity
            )
