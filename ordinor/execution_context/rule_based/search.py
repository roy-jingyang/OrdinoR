'''
A series of search-based solutions:
    - greedy descent
    - simulated annealing
'''

from collections import deque
import numpy as np

from .BaseSearch import BaseSearchMiner


class ExactSearchMiner(BaseSearchMiner):
    ...

class GreedySearchMiner(BaseSearchMiner):
    def __init__(self, 
        el, attr_spec, 
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        n_iter=10, always_move=False, n_max_move=1000
    ):
        # Initialize system parameters
        self.n_iter = n_iter
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
        while size < self.n_iter:
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
        print('Start greedy search with n_iter={}, always_move={}, n_max_move={}'.format(self.n_iter, self.always_move, self.n_max_move))

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
        n_iter=10, T0=1000, Tmin=1, alpha=1
    ):
        # Initialize system parameters
        # initialization method
        self.init_method = init_method
        # batch size to test at initialization, subject to mem size
        self.init_batch = init_batch
        # number of iterations per temperature, i.e., size of neighborhood
        self.n_iter = n_iter
        # system initial temperature
        self.T0 = T0
        # minimum temperature allowed
        self.Tmin = Tmin
        # set rate for reducing system temperature
        self.alpha = alpha

        # Initialize additional data structures (tracking visited states)
        self._visited_states = deque(maxlen=100)
        self._l_dispersal = deque(maxlen=100)
        self._l_impurity = deque(maxlen=100)

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

    def _prob_acceptance(self, E, E_next, T, 
        dis=None, dis_next=None, imp=None, imp_next=None, 
        E_best=None, dis_best=None, imp_best=None,
        **kwarg
    ):
        # NOTE: allow pr > 1 if E_next < E, to avoid the need for capping
        
        # Composite function: arithmetic mean of the same cost function
        '''
        pr = np.exp(-1 * self.T0 * 1e4 * (E_next - E) / T)
        '''

        # Composite: sum of different cost functions
        '''
        if dis_next - dis > 0:
            cost_dis = np.log(dis_next - dis + 1)
        else:
            cost_dis = -1 * np.log(dis - dis_next + 1)
        cost_imp = (imp_next - imp)
        pr = np.exp(-1 * self.T0 * (cost_dis + cost_imp) / T)
        '''

        # Lexicographical: hierarchical comparison
        delta_dis = dis_next - dis
        delta_imp = imp_next - imp
        if delta_dis <= 0 and delta_imp <= 0:
            # next '>' current
            pr = 1
        elif delta_dis > 0 and delta_imp > 0:
            # next '<' current
            pr = np.exp(-1 * self.T0 * (E_next - E) / T)
        else:
            # non-dominating pair
            theta_imp = np.abs(np.std(self._l_impurity, ddof=1))
            theta_dis = np.abs(np.std(self._l_dispersal, ddof=1))
            if np.abs(delta_imp) >= theta_imp:
                # change on impurity is significant
                pr = 1 if delta_imp <= 0 else np.exp(-1 * self.T0 * delta_imp / T)
            elif np.abs(delta_dis) >= theta_dis:
                # change on dispersal is significant
                pr = 1 if delta_dis <= 0 else np.exp(-1 * self.T0 * delta_dis / T)
            else:
                # neither changes are significant
                pr = T / self.T0

        return pr
    
    def _cooling(self, T0, k):
        return T0 - self.alpha * k

    def _search(self):
        print('Start simulated annealing search with T0={}, Tmin={}, alpha={}:'.format(self.T0, self.Tmin, self.alpha))

        #self._verify_state(self._nodes)

        T = self.T0
        ret = self._evaluate(self._nodes, self._pars)
        E, dis, imp = ret[0], ret[1], ret[2]

        # TODO:
        self._visited_states.append(self._hash_state(self._pars))
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
            # Probability of acceptance, System temperature, 
            # Dispersal, Impurity, Energy
            history = [(0, self.init_method, None, None, T, dis, imp, E)]

        while T > self.Tmin:
            k += 1
            # generate neighbors
            for i in range(self.n_iter):
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

                    # TODO: save visited states
                    id_state_next = self._hash_state(new_pars)
                    if id_state_next not in self._visited_states:
                        self._visited_states.append(id_state_next)
                        print('-' * 30 + f'{len(self._visited_states)} visited' + '-' * 30)
                        self._l_dispersal.append(dis_next)
                        self._l_impurity.append(imp_next)
                    else:
                        # Tabu: prevent moving into visited states
                        continue

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
                        '''
                        has_new_best = E < E_best
                        '''
                        delta_dis = dis - dis_best
                        delta_imp = imp - imp_best
                        if delta_dis <= 0 and delta_imp <= 0:
                            # current '>' best
                            has_new_best = True
                        elif delta_dis > 0 and delta_imp > 0:
                            # current '<' best
                            has_new_best = False
                        else:
                            theta_imp = np.abs(np.std(self._l_impurity, ddof=1))
                            theta_dis = np.abs(np.std(self._l_dispersal, ddof=1))
                            if np.abs(delta_imp) >= theta_imp:
                                has_new_best = delta_imp <= 0
                            elif np.abs(delta_dis) >= theta_dis:
                                has_new_best = delta_dis < 0
                            else:
                                has_new_best = False

                        if has_new_best:
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
            T = self._cooling(self.T0, k)
        
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

class MTSASearchMiner(SASearchMiner):
    # TODO: Multi-Objective Simulated Annealing using different temperatures
    def __init__(self,
        el, attr_spec,
        init_method='random', init_batch=1000,
        random_number_generator=None,
        print_steps=True,
        trace_history=False,
        n_iter=10, alpha=1,
        T0_dis=500, Tmin_dis=1,
        T0_imp=1000, Tmin_imp=1
    ): 
        pass
    
    def _prob_acceptance(self, E, E_next, 
        T_dis, T_imp,
        dis=None, dis_next=None, imp=None, imp_next=None, 
        E_best=None, dis_best=None, imp_best=None,
        **kwarg
    ):
        pass

    def _search(self):
        print('Start simulated annealing search with ', end='')
        print('T0 (dispersal)={}, Tmin (dispersal)={}, '.format(self.T0_dis, self.Tmin_dis), end='')
        print('T0 (impurity)={}, Tmin (impurity)={}, '.format(self.T0_imp, self.Tmin_imp), end='')
        print('alpha={}:'.format(self.alpha))
        pass
