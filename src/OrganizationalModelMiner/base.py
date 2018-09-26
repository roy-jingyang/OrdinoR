# -*- coding: utf-8 -*-

'''
This module contains the definition of the class OrganizationalModel, of which
an object would be the discovery result of organizational model mining methods
included in the sibling modules.

This module also contains the implementation of the default mining method
by Song & van der Aalst (ref. Song & van der Aalst, DSS 2008), as a simple
approach.
'''

from collections import defaultdict

class OrganizationalModel:
    '''This class provides mainly the definition of the underly data structure:
        * Resource Group ("rg")
        * Membership ("mem"): Resource Group -> PowerSet(Resource)
        * Capability ("cap"): Resource Group -> PowerSet(Execution Mode)

    In the implementation, a integer id will be assigned to each resource
    group, which will then act as an "external key".
    
    Methods related to the query (retrieval?) on an organizaitonal model are
    also defined.
    '''

    # TODO: Resource Group should be a mapping from RG id to their descriptions
    # For now the description part is not be used since we don't know how to
    # annotate a resource group.
    _rg = dict()
    _mem = dict()
    _cap = dict()
    
    # An extra python dict recording the belonging of each resource, i.e. a
    # reverse mapping of Membership ("mem") - the individual resource POV.
    _rmem = defaultdict(lambda: set())
    # An extra python dict recording the qualified groups for each execution
    # mode, i.e. a reverse mapping of Capability ("cap").
    _rcap = defaultdict(lambda: set())

    def __init__(self):
        self._rg_id = -1
    
    def add_group(self, resources, exec_modes):
        '''Add a new group into the organizational model.

        Parameters
        ----------
        resources: iterator
            The ids of resources to be added as a resource group.
        exec_modes: iterator
            The execution modes corresponding to the resources.

        Returns
        -------
        '''
        self._rg_id += 1
        self._rg[self._rg_id] = dict()

        self._mem[self._rg_id] = set()
        # two-way dict here
        for r in resources:
            self._mem[self._rg_id].add(r)
            self._rmem[r].add(self._rg_id)

        # another two-way dict here
        self._cap[self._rg_id] = set()
        for cap in exec_modes:
            self._cap[self._rg_id].add(cap)
            self._rcap[cap].add(self._rg_id)
        
        return

    def size(self):
        '''Query the size (number of organizational groups) of the model.

        Parameters
        ----------

        Returns
        -------
        int
            The number of the groups.
        '''
        return len(self._rg)

    def resources(self):
        '''Simply return all the resources involved in the model.

        Parameters
        ----------

        Returns
        -------
        set
            The set of all resources.
        '''
        return set(self._rmem.keys())

    def find_group(self, r):
        '''Query the membership (i.e. belonging to which groups) of a resource
        given its identifier.

        Parameters
        ----------
        r: 
            The identifier of a resource given.

        Returns
        -------
        list of sets
            The groups to which the queried resource belong.
        '''
        return [self._mem[rg_id] for rg_id in self._rmem[r]]

    def find_all_groups(self):
        '''Simply return all the discovered groups.

        Parameters
        ----------

        Returns
        -------
        list of sets
            The groups to which the queried resource belong.
        '''
        return [g for g in self._mem.values()]
    
    def get_candidates(self, exec_mode):
        '''Query the capability (i.e. execution modes allowed by the model) 
        of a resource given its identifier.

        Parameters
        ----------
        exec_mode: set of 3-tuples
            The identifier of a resource given.

        Returns
        -------
        list of sets
            The groups which is capable of this execution mode.
        '''
        return [self._mem[rg_id] for rg_id in self._rcap[exec_mode]]

    # IO related methods
    # TODO 
    def to_csv(self, f):
        pass

    def from_csv(self, fn):
        pass

# Note: this method does not require explicitly discovered resource profiles to
# be used.
# To stick to the original design of the method, each activity name in
# the source event log should be mapped to a single Activity Type in the
# resource log.
def default_mining(rl):
    '''
    The default mining method.

    Params:
        rl: DataFrame
            The resource log.

    Returns:
        om: OrganizationalModel object
            The discovered organizational model.
    '''

    from .mode_assignment import default_assign

    print('Applying Default Mining:')
    om = OrganizationalModel()
    for atype, events in rl.groupby('activity_type'):
        group = set(events['resource'])
        exec_modes = default_assign(group, rl)
        om.add_group(group, exec_modes)
    print('{} organizational groups discovered.'.format(om.size()))
    return om

